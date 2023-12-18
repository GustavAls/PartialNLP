import copy
import os.path
import pickle
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader, Dataset
import numpy as np
from SentimentClassifier import *
from argparse import Namespace
from PartialConstructor import PartialConstructorSwag
import utils
import time
from tqdm import tqdm
TRANSFORMER_INCOMPATIBLE_MODULES = (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d,
                                    nn.BatchNorm2d, nn.BatchNorm3d)

TRANSFORMER_COMPATIBLE_MODULES = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.Conv1d)


class SWAGExperiments:

    def __init__(self, args=None):
        # TODO set correct train and val sizes
        self.default_args = {'output_path': args.output_path,
                             'train_batch_size': args.batch_size, 'eval_batch_size': args.batch_size,'device_batch_size': args.batch_size,
                             'device': 'cuda', 'num_epochs': 1.0, 'dataset_name': args.dataset_name, 'train': True,
                             'train_size': args.train_size, 'val_size': args.val_size, 'test_size': args.test_size,  'learning_rate': 5e-05,
                             'laplace': True, 'save_strategy': 'no', 'load_best_model_at_end': False, 'no_cuda': False }

        self.default_args = Namespace(**self.default_args)
        self.default_args.model_path = args.model_path
        self.default_args.data_path = getattr(args, 'data_path', None)
        self.default_args_swag = {'n_iterations_between_snapshots': 1,
                                  'module_names': None, 'num_columns': 20, 'num_mc_samples': 50,
                                  'min_var': 1e-20, 'reduction': 'mean', 'num_classes': 2, 'optim_max_num_steps': 2 ,
                                  'max_num_steps': 2000}

        self.partial_constructor = None
        if args.subclass == "attn":
            self.num_modules = [1, 2, 3, 4, 5, 8, 11, 17]
        elif args.subclass == "mlp":
            self.num_modules = [1, 2, 3, 4, 5, 8, 11]
        # Random ramping
        else:
            self.num_modules = [1, 2, 3, 4, 5, 8, 11, 38]

        self.sentiment_classifier = None
        self.train_loader, self.trainer, self.tokenized_val, self.optimizer = (None, None, None, None)
        self.loss_fn = nn.CrossEntropyLoss()
        self.subclass = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize_sentiment_classifier(self):
        self.sentiment_classifier = prepare_sentiment_classifier(self.default_args)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader, self.trainer, self.tokenized_val = self.sentiment_classifier.prepare_laplace(
            output_path=self.default_args.output_path,
            train_bs=self.default_args.train_batch_size,
            eval_bs=self.default_args.eval_batch_size,
            dataset_name=self.default_args.dataset_name,
            train_device_batch_size=self.default_args.device_batch_size,
            eval_device_batch_size=self.default_args.device_batch_size,
            lr=self.default_args.learning_rate,
            data_path=self.default_args.data_path)


    def use_subclass_part_only(self):
        if self.subclass == 'attn':
            self.partial_constructor.set_use_only_attn()
        elif self.subclass == 'mlp':
            self.partial_constructor.set_use_only_mlp()

    def initialize_swag(self, model, **kwargs):

        kwargs = kwargs if len(kwargs) > 0 else self.default_args_swag
        self.partial_constructor = PartialConstructorSwag(model, **kwargs)
        self.use_subclass_part_only()

    def create_partial_max_operator_norm(self, num_params):
        self.partial_constructor.select_max_operator_norm(num_params)
        self.partial_constructor.select()

    def create_partial_random_ramping_construction(self, num_params):
        self.partial_constructor.select_random_percentile(num_params)
        self.partial_constructor.select()

    def create_partial_sublayer_ramping(self, percentile):
        self.partial_constructor.select_all_modules()
        self.partial_constructor.select(percentile = percentile)

    def ensure_prior_calls(self, **kwargs):

        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized")
        if 'max_num_steps' not in kwargs:
            UserWarning("Using full num epochs to train swag")

        if self.partial_constructor.module_names is None or len(self.partial_constructor.module_names) == 0:
            raise ValueError("Partial constructor has not selected any modules to make bayesian")

    def fit(self, **kwargs):
        learning_rate = kwargs.get('learning_rate') if 'learning_rate' in kwargs else self.default_args.learning_rate

        self.optimizer = torch.optim.SGD(self.partial_constructor.parameters(),
                                         lr=learning_rate, weight_decay=3e-4, momentum=0.9)

        self.ensure_prior_calls(**kwargs)

        max_num_steps = kwargs.get('max_num_steps', np.inf)
        counter = 0
        pbar = tqdm(total=max_num_steps//self.partial_constructor.number_of_iterations_bs, desc='Training SWAG')

        for epoch in range(max(int(self.default_args.num_epochs), 1)):
            for step, x in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                out = self.partial_constructor(**x)
                out.loss.backward()
                self.optimizer.step()
                self.partial_constructor.reset_params_with_mask()
                if self.partial_constructor.scheduler():
                    self.partial_constructor.snapshot()
                    pbar.update(1)

                if counter >= max_num_steps:
                    self.partial_constructor.snapshot()
                    pbar.update(pbar.total)
                    pbar.close()
                    return self.partial_constructor

                counter += 1

        pbar.update(pbar.total)
        pbar.close()
        self.partial_constructor.snapshot()
        return self.partial_constructor

    def optimize_lr(self):

        learning_rates = np.logspace(-3, -1, num=6, endpoint=True)
        learning_rates = np.array([1e-3, 1e-2, 2e-2, 1e-1])
        neg_log_likelihoods = []
        pbar = tqdm(learning_rates, desc='Optimizing Learning Rates')
        for learning_rate in pbar:
            self.fit(**{'learning_rate': learning_rate, 'max_num_steps': self.default_args_swag['optim_max_num_steps']})
            evaluator = utils.evaluate_swag(self.partial_constructor, self.trainer, self.tokenized_val)
            neg_log_likelihoods.append(evaluator.results['nll'])
            self.partial_constructor.init_new_model_for_optim(copy.deepcopy(self.trainer.model))
            pbar.update(1)

        pbar.close()
        optimimum_learning_rate = sorted(zip(neg_log_likelihoods, learning_rates))[0][1]
        return optimimum_learning_rate

    def ensure_path_existence(self, path):
        if not os.path.exists(path):
            if not os.path.exists(os.path.dirname(path)):
                raise ValueError("Make at least the folder over where the experiments are being put")
            else:
                os.mkdir(path)

    def get_num_remaining_modules(self, path, run_number):
        results_path = os.path.join(path, f"run_number_{run_number}.pkl")
        if not os.path.exists(results_path):
            return self.num_modules
        else:
            results_file = pickle.load(open(results_path, 'rb'))
            number_of_modules = list(results_file.keys())
            new_modules_to_run = sorted(list(set(self.num_modules) - set(number_of_modules)))
            return new_modules_to_run

    def random_ramping_experiment(self, run_number=0):

        results = {}
        save_path = self.default_args.output_path
        self.ensure_path_existence(save_path)
        remaining_modules = self.get_num_remaining_modules(save_path, run_number)
        if len(remaining_modules) < len(self.num_modules):
            results = pickle.load(open(os.path.join(save_path, f"run_number_{run_number}.pkl"), 'rb'))
        for number_of_modules in remaining_modules:
            print("Training with number of stochastic modules equal to", number_of_modules)
            self.initialize_sentiment_classifier()
            self.initialize_swag(copy.deepcopy(self.trainer.model))
            self.create_partial_random_ramping_construction(number_of_modules)
            print("Swag initialized")
            get_mem_nvidia()

            optimimum_learning_rate = self.optimize_lr()
            print("Optimized learning rate")
            get_mem_nvidia()

            train_kwargs = {'learning_rate': optimimum_learning_rate,
                            'max_num_steps': self.default_args_swag['max_num_steps']}
            self.partial_constructor.init_new_model_for_optim(copy.deepcopy(self.trainer.model))
            print("Model with best learning rate initialized")
            get_mem_nvidia()

            self.fit(**train_kwargs)
            print("Model trained")
            get_mem_nvidia()

            evaluator = utils.evaluate_swag(self.partial_constructor, self.trainer)
            print("Model evaluated")
            get_mem_nvidia()
            results[number_of_modules] = evaluator

            with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def sublayer_experiment_full(self, run_number = 0):
        results = {}
        save_path = self.default_args.output_path
        self.ensure_path_existence(save_path)
        self.num_modules = list(np.linspace(1, 60, num=6, endpoint=True))
        remaining_modules = self.get_num_remaining_modules(save_path, run_number)

        if len(remaining_modules) < len(self.num_modules):
            results = pickle.load(open(os.path.join(save_path, f"run_number_{run_number}.pkl"), 'rb'))

        for number_of_modules in remaining_modules:
            print("Training with number of stochastic modules equal to", number_of_modules)
            self.initialize_sentiment_classifier()
            self.default_args_swag['use_sublayer'] = True
            self.initialize_swag(copy.deepcopy(self.trainer.model))
            self.create_partial_sublayer_ramping(number_of_modules) # hereditary naming of things here
            print("Swag initialized")
            get_mem_nvidia()

            optimimum_learning_rate = self.optimize_lr()
            print("Optimized learning rate")
            get_mem_nvidia()

            train_kwargs = {'learning_rate': optimimum_learning_rate,
                            'max_num_steps': self.default_args_swag['max_num_steps']}

            self.partial_constructor.init_new_model_for_optim(copy.deepcopy(self.trainer.model))
            print("Model with best learning rate initialized")
            get_mem_nvidia()

            self.fit(**train_kwargs)
            print("Model trained")
            get_mem_nvidia()

            evaluator = utils.evaluate_swag(self.partial_constructor, self.trainer)
            print("Model evaluated")
            get_mem_nvidia()
            results[number_of_modules] = evaluator

            with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def max_norm_ramping_experiment(self, run_number=0):

        results = {}
        save_path = self.default_args.output_path
        self.ensure_path_existence(save_path)
        remaining_modules = self.get_num_remaining_modules(save_path, run_number)
        if len(remaining_modules) < len(self.num_modules):
            results = pickle.load(open(os.path.join(save_path, f"run_number_{run_number}.pkl"), 'rb'))
        for number_of_modules in remaining_modules:
            print("Training with number of stochastic modules equal to", number_of_modules)
            self.initialize_sentiment_classifier()
            self.initialize_swag(copy.deepcopy(self.trainer.model))
            self.create_partial_max_operator_norm(number_of_modules)
            print("Swag initialized")
            get_mem_nvidia()

            optimimum_learning_rate = self.optimize_lr()
            print("Optimized learning rate")
            get_mem_nvidia()

            train_kwargs = {'learning_rate': optimimum_learning_rate,
                            'max_num_steps': self.default_args_swag['max_num_steps']}

            self.partial_constructor.init_new_model_for_optim(copy.deepcopy(self.trainer.model))
            print("Model with best learning rate initialized")
            get_mem_nvidia()

            self.fit(**train_kwargs)
            print("Model trained")
            get_mem_nvidia()

            evaluator = utils.evaluate_swag(self.partial_constructor, self.trainer)
            print("Model evaluated")
            get_mem_nvidia()
            results[number_of_modules] = evaluator

            with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_random_ramping_experiments(args):
    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    exp_args = {'model_path': model_path,
                'dataset_name': args.dataset_name,
                'data_path': data_path,
                'output_path': args.output_path,
                'batch_size': args.batch_size,
                'train_size': args.train_size,
                'subclass': args.subclass,
                'val_size': args.val_size,
                'test_size': args.test_size}

    exp_args = Namespace(**exp_args)
    swag_exp = SWAGExperiments(args=exp_args)
    swag_exp.random_ramping_experiment(args.run_number)

def run_max_norm_ramping_experiments(args):
    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    exp_args = {'model_path': model_path,
                'dataset_name': args.dataset_name,
                'data_path': data_path,
                'output_path': args.output_path,
                'batch_size': args.batch_size,
                'train_size': args.train_size,
                'subclass': args.subclass,
                'val_size': args.val_size,
                'test_size': args.test_size}

    exp_args = Namespace(**exp_args)
    swag_exp = SWAGExperiments(args=exp_args)
    swag_exp.max_norm_ramping_experiment(args.run_number)


def run_max_norm_ramping_only_subclass(args):
    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    exp_args = {'model_path': model_path,
                'dataset_name': args.dataset_name,
                'data_path': data_path,
                'output_path': args.output_path,
                'batch_size': args.batch_size,
                'train_size': args.train_size,
                'subclass': args.subclass,
                'val_size': args.val_size,
                'test_size': args.test_size}

    exp_args = Namespace(**exp_args)
    swag_exp = SWAGExperiments(args=exp_args)
    swag_exp.subclass = args.subclass
    swag_exp.max_norm_ramping_experiment(args.run_number)

def run_sublayer_experiment(args):
    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    exp_args = {'model_path': model_path,
                'dataset_name': args.dataset_name,
                'data_path': data_path,
                'output_path': args.output_path,
                'batch_size': args.batch_size,
                'train_size': args.train_size,
                'subclass': args.subclass,
                'val_size': args.val_size,
                'test_size': args.test_size}

    exp_args = Namespace(**exp_args)
    swag_exp = SWAGExperiments(args=exp_args)
    swag_exp.subclass = args.subclass
    swag_exp.sublayer_experiment_full(args.run_number)



def get_mem_nvidia():
    if torch.cuda.is_available():
        import pynvml as nvml

        def bytes_to_gb(bytes):
            return round(bytes / 1024**3, 4)

        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()

        for i in range(device_count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"Total memory (gb): {bytes_to_gb(info.total)}, Free memory (gb): {bytes_to_gb(info.free)}, Used memory (gb): {bytes_to_gb(info.used)}")
            except:
                print("Can't get memory info for device", i)
                continue
        nvml.nvmlShutdown()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run training and or evaluation of Sentiment Classifier with swag"
    )
    parser.add_argument("--run_number", type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='imdb')
    parser.add_argument('--experiment', type=str, default='')
    parser.add_argument('--data_path', type = str, default='')
    parser.add_argument('--output_path', type = str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--subclass', type = str, default='')
    parser.add_argument('--train_size', type=int, default=1)
    parser.add_argument('--val_size', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=1)

    args = parser.parse_args()

    if args.experiment == 'random_ramping':
        run_random_ramping_experiments(args)
    if args.experiment == 'operator_norm_ramping':
        run_max_norm_ramping_experiments(args)
    if args.experiment == 'operator_norm_ramping_subclass':
        if args.subclass:
            run_max_norm_ramping_only_subclass(args)
    if args.experiment == 'sublayer':
        run_sublayer_experiment(args)
