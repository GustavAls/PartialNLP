import copy

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader, Dataset
import numpy as np
from SentimentClassifier import *
from argparse import Namespace
from PartialConstructor import PartialConstructorSwag
import utils
TRANSFORMER_INCOMPATIBLE_MODULES = (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d,
                                    nn.BatchNorm2d, nn.BatchNorm3d)

TRANSFORMER_COMPATIBLE_MODULES = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.Conv1d)


class SWAGExperiments:

    def __init__(self, args = None):
        # TODO set correct train and val sizes
        self.default_args = {'output_path': 'C:\\Users\\45292\\Documents\\Master\\SentimentClassification',
                             'train_batch_size': 1, 'eval_batch_size': 1, 'device': 'cpu', 'num_epochs': 1.0,
                             'dataset_name': 'imdb',
                             'train': True, 'train_size': None, 'test_size': None, 'device_batch_size': 1,
                             'learning_rate': 5e-05, 'seed': 0, 'train_size': 2, 'test_size': 2,
                             'laplace': True, 'swag': True, 'save_strategy': 'no',
                             'load_best_model_at_end': False, 'no_cuda': False}

        self.default_args = Namespace(**self.default_args)
        self.default_args.model_path = args.model_path
        self.default_args_swag = {'n_iterations_between_snapshots': 100,
                                  'module_names': None, 'num_columns': 3, 'num_mc_samples': 50,
                                  'min_var': 1e-20, 'reduction': 'mean', 'num_classes': 2,'optim_max_num_steps': 10,
                                  'max_num_steps': 10}

        self.partial_constructor = None

        self.sentiment_classifier = None
        self.train_loader, self.trainer, self.tokenized_val, self.optimizer = (None, None, None, None)
        self.loss_fn = nn.CrossEntropyLoss()


    def initialize_sentiment_classifier(self):
        self.sentiment_classifier = prepare_sentiment_classifier(self.default_args)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loader, self.trainer, self.tokenized_val = self.sentiment_classifier.prepare_laplace(
            output_path=self.default_args.output_path,
            train_bs=self.default_args.train_batch_size,
            eval_bs=self.default_args.eval_batch_size,
            dataset_name=self.default_args.dataset_name,
            device_batch_size=self.default_args.device_batch_size,
            lr=self.default_args.learning_rate)


    def initialize_swag(self, model, **kwargs):

        kwargs = kwargs if len(kwargs) > 0 else self.default_args_swag
        self.partial_constructor = PartialConstructorSwag(model, **kwargs)

    def create_partial_random_ramping_construction(self, num_params):
        self.partial_constructor.select_random_percentile(num_params)
        self.partial_constructor.select()

    def ensure_prior_calls(self, **kwargs):

        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized")
        if 'max_num_steps' not in kwargs:
           UserWarning("Using full num epochs to train swag")

        if self.partial_constructor.module_names is None or len(self.partial_constructor.module_names) == 0:
            raise ValueError("Partial constructor has not selected any modules to make bayesian")

    def fit(self, **kwargs):
        learning_rate = kwargs.get('learning_rate') if 'learning_rate' in kwargs else self.default_args.learning_rate

        self.optimizer = torch.optim.SGD(self.partial_constructor.model.parameters(),
                                         lr = learning_rate)

        self.ensure_prior_calls(**kwargs)

        max_num_steps = kwargs.get('max_num_steps', np.inf)
        counter = 0
        for epoch in range(max(int(self.default_args.num_epochs), 1)):
            for step, x in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                out = self.partial_constructor(**x)
                out.loss.backward()
                self.optimizer.step()

                if self.partial_constructor.scheduler():
                    self.partial_constructor.snapshot()

                if counter == max_num_steps:
                    return self.partial_constructor

                counter += 1

        self.partial_constructor.snapshot()
        return self.partial_constructor

    def optimize_lr(self):

        learning_rates = np.logspace(-3, -1, num=6, endpoint=True)
        neg_log_likelihoods = []
        for learning_rate in learning_rates:
            self.fit(**{'learning_rate': learning_rate, 'max_num_steps': self.default_args_swag['optim_max_num_steps']})
            evaluator = utils.evaluate_swag(self.partial_constructor, self.trainer, self.tokenized_val)
            neg_log_likelihoods.append(evaluator.results['nll'])
            self.partial_constructor.init_new_model_for_optim(copy.deepcopy(self.trainer.model))

        optimimum_learning_rate = sorted(zip(neg_log_likelihoods, learning_rates))[0][1]
        return optimimum_learning_rate

    def ensure_path_existence(self, path):
        if not os.path.exists(path):
            if not os.path.exists(os.path.dirname(path)):
                raise ValueError("Make at least the folder over where the experiments are being put")
            else:
                os.mkdir(path)

    def random_ramping_experiment(self, run_number = 0):

        num_modules = [1, 2, 3, 4, 5, 8, 11, 17, 28, 38]
        results = {}
        for number_of_modules in num_modules:
            self.initialize_sentiment_classifier()
            self.initialize_swag(copy.deepcopy(self.trainer.model))
            self.create_partial_random_ramping_construction(number_of_modules)
            optimimum_learning_rate = self.optimize_lr()

            train_kwargs = {'learning_rate': optimimum_learning_rate,
                            'max_num_steps': self.default_args_swag['max_num_steps']}

            self.partial_constructor.init_new_model_for_optim(copy.deepcopy(self.trainer.model))
            self.fit(**train_kwargs)

            evaluator = utils.evaluate_swag(self.partial_constructor, self.trainer)
            results[number_of_modules] = evaluator

        save_path = os.path.join(self.default_args.output_path, 'random_module_ramping')
        self.ensure_path_existence(save_path)
        with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_random_ramping_experiments(args):

    model_paths = [r"C:\Users\45292\Documents\Master\SentimentClassification\checkpoint-782"]
    exp_args = {'model_path': model_paths[args.run_number],
            'dataset_name': args.dataset_name}
    exp_args = Namespace(**exp_args)
    swag_exp = SWAGExperiments(args = exp_args)
    swag_exp.random_ramping_experiment(args.run_number)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run training and or evaluation of Sentiment Classifier with swag"
    )
    parser.add_argument("--run_number", type=int, default=0)
    parser.add_argument('--dataset_name', type = str, default='imdb')
    parser.add_argument('--experiment', type = str, default = '')
    args = parser.parse_args()


    if args.experiment == 'random_ramping':
        run_random_ramping_experiments(args)




