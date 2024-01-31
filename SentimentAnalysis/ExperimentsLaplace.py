import ast
import copy
import os
import pickle

import torch
from datasets import load_dataset
import datasets
import evaluate
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
import numpy as np
import argparse
import yaml
from argparse import Namespace
from transformers.models.llama import LlamaForSequenceClassification
import importlib
# Heavily based on: https://huggingface.co/blog/sentiment-analysis-python, and
# https://huggingface.co/docs/transformers/tasks/sequence_classification
# from  import laplace_partial as lp
# from laplace_lora.laplace_partial.utils import ModuleNameSubnetMask
import laplace_partial as lp
from PartialConstructor import PartialConstructor, PartialConstructorSwag, Truncater, Extension
import torch.nn as nn
from Laplace.laplace import Laplace
from torch.utils.data import Dataset, DataLoader
import utils
from SentimentClassifier import construct_laplace, prepare_sentiment_classifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
from NLIClassifier import prepare_nli_classifier

EXPERIMENTS = ['last_layer',
               'random_ramping',
               'operator_norm_ramping',
               'operator_norm_ramping_subclass',
               'sublayer_full',
               'sublayer_predefined',
               'nli_last_layer',
               'nli_random_ramping',
               'nli_operator_norm_ramping',
               'nli_sublayer_full',
               'nli_sublayer_predefined']

class LaplaceExperiments:
    def __init__(self, args, nli = False):
        self.default_args = {'output_path': args.output_path,
                             'train_batch_size': args.train_batch_size, 'eval_batch_size': args.eval_batch_size,
                             'train_device_batch_size': args.train_batch_size, 'eval_device_batch_size': args.eval_batch_size,
                             'device': 'cuda', 'num_epochs': 1.0, 'dataset_name': args.dataset_name,
                             'train': True, 'train_size': args.train_size, 'val_size': args.val_size,
                             'test_size': args.test_size, 'learning_rate': 5e-05,
                             'laplace': True, 'save_strategy': 'no',
                             'load_best_model_at_end': False, 'no_cuda': False}

        self.base_full_prior = 1e4
        self.base_min_prior = 1
        self.num_params = 0
        self.num_stoch_params = 0
        self.args = args
        self.module_names = None
        self.include_last_layer = args.include_last_layer
        self.last_layer_full = args.last_layer_full
        # self.num_modules = [1, 2, 3, 4, 5, 8, 11, 17, 28, 38]
        # Memory consideration
        if args.subclass == 'attn':
            self.num_modules = [1, 2, 3, 4, 5, 8, 11, 13, 17]
        else:
            self.num_modules = [1, 2, 3, 4, 5, 8, 11]
        default_args = Namespace(**self.default_args)
        self.default_args = default_args
        default_args.model_path = args.model_path

        # NLI classifier or sentiment classifier
        if nli:
            self.classifier = prepare_nli_classifier(default_args)
        else:
            self.classifier = prepare_sentiment_classifier(default_args)

        self.subclass = args.subclass
        self.batched_modules_for_batch_grad = args.num_batched_modules

        self.minimum_prior, self.maximum_prior, self.best_nll = 1e-1, 1e5, np.inf
        self.train_loader, self.trainer, self.tokenized_val = self.classifier.prepare_laplace(
            output_path=default_args.output_path,
            train_bs=default_args.train_batch_size,
            eval_bs=default_args.eval_batch_size,
            dataset_name=default_args.dataset_name,
            train_device_batch_size=default_args.train_device_batch_size,
            eval_device_batch_size=default_args.eval_device_batch_size,
            lr=default_args.learning_rate,
            data_path=args.data_path,
            run=args.run_number)


        if not isinstance(self.classifier.model, Extension):
            self.model = Extension(self.classifier.model)
        else:
            self.model = self.classifier.model

    def create_partial_sublayer_full_model(self, percentile = 0.1):
        partial_constructor = self.use_subclass_part_only(PartialConstructor(self.model))
        partial_constructor.select_all_modules()
        partial_constructor.select_sublayer_kfac(percentile=percentile)
        partial_constructor.select()
        setattr(self.model, 'batched_modules', self.batched_modules_for_batch_grad)
        self.num_stoch_params = partial_constructor.get_num_stochastic_parameters()
        self.num_params = partial_constructor.get_num_params()
        self.module_names = partial_constructor.module_names

    def create_partial_sublayer_specific_modules(self, module_names, percentile = 0.1):
        partial_constructor = PartialConstructor(self.model)
        partial_constructor.select_predifined_modules(module_names)
        partial_constructor.select_sublayer_kfac(percentile=percentile)
        partial_constructor.select()
        self.num_stoch_params = partial_constructor.get_num_stochastic_parameters()
        self.num_params = partial_constructor.get_num_params()
        self.module_names = partial_constructor.module_names

    def create_partial_random_ramping_construction(self, num_params):
        partial_constructor = self.use_subclass_part_only(PartialConstructor(self.model))
        partial_constructor.select_random_percentile(num_params)
        partial_constructor.select()
        self.num_stoch_params = partial_constructor.get_num_stochastic_parameters()
        self.num_params = partial_constructor.get_num_params()
        self.module_names = partial_constructor.module_names

    def create_partial_last_layer(self):
        partial_constructor = PartialConstructor(self.model)
        partial_constructor.select_last_layer()
        partial_constructor.select()
        self.num_stoch_params = partial_constructor.get_num_stochastic_parameters()
        self.num_params = partial_constructor.get_num_params()
        self.module_names = partial_constructor.module_names


    def select_last_layer_with_other_methods(self):
        if self.include_last_layer:
            self.module_names = self.module_names[:-1]
            partial_constructor = PartialConstructor(self.model)
            partial_constructor.select_last_layer()
            partial_constructor.module_names += self.module_names
            partial_constructor.select()
            setattr(self.model, 'batched_modules', self.batched_modules_for_batch_grad)
            self.num_stoch_params = partial_constructor.get_num_stochastic_parameters()
            self.num_params = partial_constructor.get_num_params()
            self.module_names = partial_constructor.module_names

    def create_partial_max_norm_ramping(self, num_params, use_minimum = False):
        partial_constructor = self.use_subclass_part_only(PartialConstructor(self.model))
        if use_minimum:
            partial_constructor.select_min_operator_norm(num_params)
        else:
            partial_constructor.select_max_operator_norm(num_params)
        partial_constructor.select()
        setattr(self.model, 'batched_modules', self.batched_modules_for_batch_grad)
        self.num_stoch_params = partial_constructor.get_num_stochastic_parameters()
        self.num_params = partial_constructor.get_num_params()
        self.module_names = partial_constructor.module_names

    def use_subclass_part_only(self, partial_constructor):

        if self.subclass == 'attn':
            partial_constructor.set_use_only_attn()
        elif self.subclass == 'mlp':
            partial_constructor.set_use_only_mlp()

        return partial_constructor

    def fit_laplace(self, prior_precision=1.0, train_loader = None):
        if self.last_layer_full:
            la = lp.Laplace(self.model, 'classification',
                            hessian_structure='full'
                            )
        else:
            la = lp.Laplace(self.model, 'classification',
                            subset_of_weights='all',  # Deprecated
                            hessian_structure='kron',
                            prior_precision=prior_precision)

        train_loader = self.train_loader if train_loader is None else train_loader
        la.fit(train_loader)
        return la

    def train_and_predict_new_prior(self, priors, nlls):

        if isinstance(priors, list):
            prior = np.array(priors)[:, None]
        if isinstance(nlls, list):
            nll = np.array(nlls)[:, None]

        prior = prior / max(prior)
        prior = prior - min(prior)
        prediction_space = np.linspace(0, 1, 1000).reshape(-1, 1)
        choice_space = np.linspace(self.minimum_prior, self.maximum_prior, 1000)
        kernel = RBF()
        gaussian_process = GaussianProcessRegressor(kernel = kernel, normalize_y=True).fit(prior, nll)
        predictions, std = gaussian_process.predict(prediction_space, return_std=True)
        minimum_prediction = choice_space[np.argmin(predictions - 1.96 *std)]

        return minimum_prediction

    def optimize_prior_precision(self, num_steps = 7, use_uninformed = False):
        if use_uninformed:
            UserWarning("Optimizing called but using uninformed prior")
            percentage_of_stoch_params = self.num_stoch_params/self.num_params
            coeff = (self.base_full_prior - self.base_min_prior)
            prior = self.base_min_prior + coeff * percentage_of_stoch_params
            la = self.fit_laplace(prior_precision=prior)
            return la, prior

        random_subset_indices_val = torch.randperm(len(self.tokenized_val))[:int(len(self.tokenized_val) * 0.5)]
        new_tokenized_val = datasets.Dataset.from_dict(self.tokenized_val[random_subset_indices_val])

        la = self.fit_laplace(train_loader=self.train_loader)
        prior_precisions = np.logspace(-1, 3, num=num_steps, endpoint=True)
        pbar = tqdm(prior_precisions, desc='Training prior precision')
        overall_results = []

        for prior in pbar:
            la.prior_precision = torch.ones_like(la.prior_precision)*prior
            evaluator = utils.evaluate_laplace(la, self.trainer, new_tokenized_val)
            overall_results.append((evaluator.results['nll'], prior))
            pbar.set_postfix({'nll': evaluator.results['nll'], 'prior': prior})

        best_prior = sorted(overall_results)[0][1]
        la.prior_precision = torch.ones_like(la.prior_precision)*best_prior

        return la, best_prior


    def optimize_prior_precision_v2(self, num_steps=7, dataloader_size = 0.2, use_uninformed = False):

        if use_uninformed:
            UserWarning("Optimizing called but using uninformed prior")
            percentage_of_stoch_params = self.num_stoch_params/self.num_params
            coeff = (self.base_full_prior - self.base_min_prior)
            prior = self.base_min_prior + coeff * percentage_of_stoch_params
            la = self.fit_laplace(prior_precision=prior)
            return la, prior

        from torch.utils.data import DataLoader, Subset
        if dataloader_size < 1:
            dataloader_size = int(dataloader_size * len(self.train_loader))
        elif dataloader_size == 1:
            dataloader_size = len(self.train_loader)

        random_subset_indices_train = torch.randperm(len(self.train_loader.dataset))[:dataloader_size]
        random_subset_indices_val = torch.randperm(len(self.tokenized_val))[:int(len(self.tokenized_val) * 0.5)]
        new_dataloader = utils.get_smaller_dataloader(self.train_loader, random_subset_indices_train)
        new_tokenized_val = datasets.Dataset.from_dict(self.tokenized_val[random_subset_indices_val])

        # new_dataloader = self.train_loader

        prior_precisions = np.logspace(-1, 3, num=num_steps, endpoint=True)
        overall_results = []
        pbar = tqdm(prior_precisions, desc='Training prior precision')
        for prior in pbar:
            la = self.fit_laplace(prior, train_loader = new_dataloader)
            evaluator = utils.evaluate_laplace(la, self.trainer, new_tokenized_val)
            overall_results.append((evaluator.results['nll'], prior))
            pbar.set_postfix({'nll': evaluator.results['nll'], 'prior': prior})
        best_prior = sorted(overall_results)[0][1]

        la = self.fit_laplace(prior_precision=best_prior)
        return la, best_prior

    def optimize_prior_precision_(self, num_steps=7, use_uninformed = False):

        self.best_nll = np.inf
        negative_log_likelihoods = []
        priors = [self.minimum_prior, self.maximum_prior]
        if use_uninformed:
            UserWarning("Optimizing called but using uninformed prior")
            percentage_of_stoch_params = self.num_stoch_params/self.num_params
            coeff = (self.base_full_prior - self.base_min_prior)
            prior = self.base_min_prior + coeff * percentage_of_stoch_params
            la = self.fit_laplace(prior_precision=prior)
            return la, prior

        la = self.fit_laplace(prior_precision=self.minimum_prior)
        best_la = copy.deepcopy(la)
        evaluator = utils.evaluate_laplace(la, self.trainer, self.tokenized_val)
        negative_log_likelihoods.append(evaluator.results['nll'])
        self.best_nll = negative_log_likelihoods[-1]

        la = self.fit_laplace(prior_precision=self.maximum_prior)
        evaluator = utils.evaluate_laplace(la, self.trainer, self.tokenized_val)
        negative_log_likelihoods.append(evaluator.results['nll'])

        if negative_log_likelihoods[-1] < self.best_nll:
            best_la = copy.deepcopy(la)
            self.best_nll = negative_log_likelihoods[-1]

        for step in range(num_steps):
            next_point = self.train_and_predict_new_prior(priors, negative_log_likelihoods)
            priors.append(next_point.item())

            la = self.fit_laplace(prior_precision=priors[-1])
            evaluator = utils.evaluate_laplace(la, self.trainer, self.tokenized_val)
            negative_log_likelihoods.append(evaluator.results['nll'])
            if negative_log_likelihoods[-1] < self.best_nll:
                best_la = copy.deepcopy(la)
                self.best_nll = negative_log_likelihoods[-1]

        return best_la

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
            results_key = next(iter(results_file.keys()))
            number_of_modules = list(results_file[results_key].keys())
            new_modules_to_run = sorted(list(set(self.num_modules) - set(number_of_modules)))
            return new_modules_to_run

    def get_num_remaining_percentiles(self, path, run_number):
        results_path = os.path.join(path, f"run_number_{run_number}.pkl")
        if not os.path.exists(results_path):
            return self.percentiles
        else:
            results_file = pickle.load(open(results_path, 'rb'))
            results_key = next(iter(results_file.keys()))
            num_percentiles = list(results_file[results_key].keys())
            num_perc_float = [float(num) for num in num_percentiles]
            new_percentiles_to_run = sorted(list(set(self.percentiles) - set(num_perc_float)))
            return new_percentiles_to_run


    def map_evaluation(self, run_number = 0):

        save_path = self.args.output_path
        evaluator = utils.evaluate_map(self.model, self.trainer)
        results = {'results': evaluator}

        with open(os.path.join(save_path, f'run_number_{run_number}_map.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def random_ramping_experiment(self, run_number = 0, use_uninformed = False):

        print("Running random ramping experiment on ", self.default_args.dataset_name)
        results = {'results': {}, 'module_selection': {}}

        save_path = self.args.output_path
        self.ensure_path_existence(save_path)
        remaining_modules = self.get_num_remaining_modules(save_path, run_number)
        if len(remaining_modules) < len(self.num_modules):
            results = pickle.load(open(os.path.join(save_path, f"run_number_{run_number}.pkl"), 'rb'))
        for num_modules in remaining_modules:
            self.create_partial_random_ramping_construction(num_modules)
            self.select_last_layer_with_other_methods()
            la, prior = self.optimize_prior_precision(num_steps=self.args.num_optim_steps, use_uninformed = use_uninformed)
            evaluator = utils.evaluate_laplace(la, self.trainer)
            evaluator.results['prior_precision'] = prior
            results['results'][num_modules] = copy.deepcopy(evaluator)
            results['module_selection'][num_modules] = copy.deepcopy(self.module_names)

            with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def max_norm_ramping_experiment(self, run_number = 0, use_uninformed = False, use_minimum = False):

        print("Running operator norm ramping experiment on ", self.default_args.dataset_name)
        results = {'results': {}, 'module_selection': {}}
        save_path = self.args.output_path
        self.ensure_path_existence(save_path)
        remaining_modules = self.get_num_remaining_modules(save_path, run_number)

        if len(remaining_modules) < len(self.num_modules):
            results = pickle.load(open(os.path.join(save_path, f"run_number_{run_number}.pkl"), 'rb'))

        for num_modules in remaining_modules:
            self.create_partial_max_norm_ramping(num_modules, use_minimum)
            self.select_last_layer_with_other_methods()
            la, prior = self.optimize_prior_precision(num_steps=self.args.num_optim_steps, use_uninformed = use_uninformed)
            evaluator = utils.evaluate_laplace(la, self.trainer)
            evaluator.results['prior_precision'] = prior
            results['results'][num_modules] = copy.deepcopy(evaluator)
            results['module_selection'][num_modules] = copy.deepcopy(self.module_names)

            with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_sublayer_ramping_experiment(self, run_number, use_uninformed=True, predefined_percentiles = False, percentiles_list=None):
        print("Running sublayer full ramping experiment on ", self.default_args.dataset_name)
        results = {'results': {}, 'percentile': {}}
        save_path = self.args.output_path
        self.ensure_path_existence(save_path)
        # This is the percentiles subsampled in input and output dimensions of each module, which means
        # that the total number of parameters sampled scales quadratically with self.percentiles

        if predefined_percentiles:
            self.percentiles = np.linspace(predefined_percentiles[0],
                                           predefined_percentiles[1],
                                           predefined_percentiles[2], endpoint=True)
        elif percentiles_list:
            self.percentiles = percentiles_list
        else:
            self.percentiles = np.linspace(1, 30, 6, endpoint=True)
        remaining_percentiles = self.get_num_remaining_percentiles(save_path, run_number)

        if len(remaining_percentiles) < len(self.percentiles):
            results = pickle.load(open(os.path.join(save_path, f"run_number_{run_number}.pkl"), 'rb'))

        for percentile in remaining_percentiles:
            print(f"Running percentile {percentile} out of {self.percentiles}")
            self.create_partial_sublayer_full_model(percentile=percentile)

            la, prior = self.optimize_prior_precision(self.args.num_optim_steps, use_uninformed=use_uninformed)
            evaluator = utils.evaluate_laplace(la, self.trainer)
            evaluator.results['prior_precision'] = prior
            results['results'][str(percentile)] = copy.deepcopy(evaluator)
            with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_sublayer_ramping_predefined_modules(self, run_number, modules, use_uninformed=False):
        print("Running sublayer predefined modules ramping experiment on ", self.default_args.dataset_name)
        save_path = self.args.output_path
        self.ensure_path_existence(save_path)
        self.percentiles = np.linspace(10, 90, 9)
        results = {'results': {}, 'percentile': {}}
        remaining_percentiles = self.get_num_remaining_percentiles(save_path, run_number)

        if len(remaining_percentiles) < len(self.percentiles):
            results = pickle.load(open(os.path.join(save_path, f"run_number_{run_number}.pkl"), 'rb'))

        for percentile in remaining_percentiles:
            if self.include_last_layer:
                modules += ['model.classifier']

            self.create_partial_sublayer_specific_modules(module_names=modules, percentile=percentile)

            la, prior = self.optimize_prior_precision(self.args.num_optim_steps, use_uninformed=use_uninformed)
            evaluator = utils.evaluate_laplace(la, self.trainer)
            evaluator.results['prior_precision'] = prior
            results['results'][str(percentile)] = copy.deepcopy(evaluator)
            with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def last_layer_experiment(self, run_number, use_uninformed = False):
        print("Running last layer experiment on ", self.default_args.dataset_name)
        results = {'results': {}, 'module_selection': {}}
        save_path = self.args.output_path
        self.ensure_path_existence(save_path)

        self.create_partial_last_layer()
        la, prior = self.optimize_prior_precision(num_steps=self.args.num_optim_steps, use_uninformed=use_uninformed)
        evaluator = utils.evaluate_laplace(la, self.trainer)
        evaluator.results['prior_precision'] = prior
        results['results'] = copy.deepcopy(evaluator)
        results['module_selection'] = copy.deepcopy(self.module_names)

        with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_map_eval(args, nli= False):
    for run in range(0, 5):
        data_path = args.data_path
        data_path = os.path.join(data_path, f"run_{run}")
        output_path = args.output_path
        model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

        model_path = os.path.join(data_path, model_ext_path)
        args.model_path = model_path
        la_args = {'model_path': model_path,
                   'num_optim_steps': 7,
                   'data_path': data_path,
                   'run_number': run,
                   'output_path': output_path,
                   'train_batch_size': args.train_batch_size,
                   'eval_batch_size': args.eval_batch_size,
                   'dataset_name': args.dataset_name,
                   'subclass': args.subclass,
                   'train_size': args.train_size,
                   'val_size': args.val_size,
                   'test_size': args.test_size,
                   'num_batched_modules': args.num_batched_modules,
                   'include_last_layer': args.include_last_layer,
                   'last_layer_full': args.last_layer_full
                   }

        la_args = Namespace(**la_args)
        lap_exp = LaplaceExperiments(args=la_args, nli=nli)
        lap_exp.map_evaluation(run)

def run_random_ramping_experiments(args, nli = False):

    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    la_args = {'model_path': model_path,
               'num_optim_steps': 7,
               'data_path': data_path,
               'run_number': args.run_number,
               'output_path': args.output_path,
               'train_batch_size' : args.train_batch_size,
               'eval_batch_size' : args.eval_batch_size,
               'dataset_name': args.dataset_name,
               'subclass': args.subclass,
               'train_size': args.train_size,
               'val_size': args.val_size,
               'test_size': args.test_size,
               'num_batched_modules': args.num_batched_modules,
               'include_last_layer': args.include_last_layer,
               'last_layer_full': args.last_layer_full
               }

    la_args = Namespace(**la_args)
    lap_exp = LaplaceExperiments(args = la_args, nli=nli)
    lap_exp.random_ramping_experiment(args.run_number, args.uninformed_prior)


def run_max_norm_ramping_experiments(args, nli = False):

    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    la_args = {'model_path': model_path,
               'num_optim_steps': 7,
               'data_path': data_path,
               'run_number': args.run_number,
               'output_path': args.output_path,
               'train_batch_size' : args.train_batch_size,
               'eval_batch_size' : args.eval_batch_size,
               'dataset_name': args.dataset_name,
               'subclass': args.subclass,
               'train_size': args.train_size,
               'val_size': args.val_size,
               'test_size': args.test_size,
               'num_batched_modules': args.num_batched_modules,
               'include_last_layer': args.include_last_layer,
               'last_layer_full': args.last_layer_full
               }

    la_args = Namespace(**la_args)
    lap_exp = LaplaceExperiments(args = la_args, nli = nli)
    lap_exp.max_norm_ramping_experiment(args.run_number, args.uninformed_prior, args.minimum_norm)

def run_max_norm_ramping_only_subclass(args):
    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    la_args = {
        'model_path': model_path,
        'num_optim_steps': 7,
        'data_path': data_path,
        'run_number': args.run_number,
        'output_path': args.output_path,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'dataset_name': args.dataset_name,
        'subclass': args.subclass,
        'train_size': args.train_size,
        'val_size': args.val_size,
        'test_size': args.test_size,
        'num_batched_modules': args.num_batched_modules,
        'include_last_layer': args.include_last_layer,
        'last_layer_full': args.last_layer_full
    }

    la_args = Namespace(**la_args)
    lap_exp = LaplaceExperiments(args = la_args)
    lap_exp.subclass = args.subclass
    lap_exp.max_norm_ramping_experiment(args.run_number, args.uninformed_prior, args.minimum_norm)


def run_last_layer(args, run_number = 0, nli = False):
    data_path = os.path.join(args.data_path, f"run_{run_number}")
    output_path = os.path.join(args.output_path, f"run_{run_number}")
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]
    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path

    la_args = {'model_path': model_path,
               'num_optim_steps': 7,
               'data_path': data_path,
               'run_number': run_number,
               'output_path': output_path,
               'train_batch_size': args.train_batch_size,
               'eval_batch_size': args.eval_batch_size,
               'dataset_name': args.dataset_name,
               'subclass': args.subclass,
               'train_size': args.train_size,
               'val_size': args.val_size,
               'test_size': args.test_size,
               'num_batched_modules': args.num_batched_modules,
               'include_last_layer': args.include_last_layer,
               'last_layer_full': args.last_layer_full
               }

    la_args = Namespace(**la_args)
    lap_exp = LaplaceExperiments(args=la_args, nli=nli)
    lap_exp.last_layer_experiment(args.run_number, args.uninformed_prior)


def run_sublayer_ramping_predefined_modules(args, nli = False):
    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    if not os.path.exists(args.module_names_path):
        raise ValueError("For sublayer experiment with predifined modules, you need to specificy a path to"
                         "a pickle containing the desired modules for each of the runs")

    module_names = pickle.load(open(os.path.join(args.module_names_path, "module_names_2_modules.pkl"), 'rb'))
    module_names = module_names[args.run_number]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    la_args = {'model_path': model_path,
               'num_optim_steps': 7,
               'data_path': data_path,
               'run_number': args.run_number,
               'output_path': args.output_path,
               'train_batch_size': args.train_batch_size,
               'eval_batch_size': args.eval_batch_size,
               'dataset_name': args.dataset_name,
               'subclass': args.subclass,
               'train_size': args.train_size,
               'val_size': args.val_size,
               'test_size': args.test_size,
               'num_batched_modules': args.num_batched_modules,
               'include_last_layer': args.include_last_layer,
               'last_layer_full': args.last_layer_full
               }

    la_args = Namespace(**la_args)
    lap_exp = LaplaceExperiments(args=la_args, nli=nli)
    lap_exp.run_sublayer_ramping_predefined_modules(args.run_number,module_names, args.uninformed_prior)

def run_sublayer_ramping_full(args, nli=False):

    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    la_args = {'model_path': model_path,
               'num_optim_steps': 7,
               'data_path': data_path,
               'run_number': args.run_number,
               'output_path': args.output_path,
               'train_batch_size' : args.train_batch_size,
               'eval_batch_size' : args.eval_batch_size,
               'dataset_name': args.dataset_name,
               'subclass': args.subclass,
               'train_size': args.train_size,
               'val_size': args.val_size,
               'test_size': args.test_size,
               'num_batched_modules': args.num_batched_modules,
               'include_last_layer': args.include_last_layer,
               'last_layer_full': args.last_layer_full
               }

    la_args = Namespace(**la_args)
    lap_exp = LaplaceExperiments(args=la_args, nli=nli)
    lap_exp.run_sublayer_ramping_experiment(args.run_number, args.uninformed_prior, args.percentile_range, args.percentile_list)

def sequential_last_layer(args, nli = False):
    num_runs = 5
    for run in range(num_runs):
        run_last_layer(args, run, nli=nli)

def parse_percentile_ramping_specification(args):

    if args.percentile_range:
        splitted = args.percentile_range.split()
        if len(splitted) != 3:
            raise ValueError("--percentile_range must be of the format 'start end num_points'")

        try:
            start = float(splitted[0])
            end = float(splitted[1])
            num_points = int(splitted[2])
        except:
            raise ValueError("each element in --percentile_range must be castable to float, except last which is int")

        args.percentile_range = [start, end, num_points]

    return args

def parse_percentile_list(args):
    if args.percentile_list:
        splitted = args.percentile_list.split()
        try:
            percentiles = [float(num) for num in splitted]
        except:
            raise ValueError("each element in --percentile_list must be castable to float")

        args.percentile_list = percentiles
    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run training and or evaluation of Sentiment Classifier"
    )
    parser.add_argument("--run_number", type=int, default=0)
    parser.add_argument('--dataset_name', type = str, default='imdb')
    parser.add_argument('--uninformed_prior', type = ast.literal_eval, default=False)
    parser.add_argument('--experiment', type = str, default = '')
    parser.add_argument('--data_path', type = str, default='')
    parser.add_argument('--model_path', type = str, default='')
    parser.add_argument('--output_path', type = str, default='')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--train_size', type=int, default=1)
    parser.add_argument('--val_size', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=1)
    parser.add_argument('--subclass', type = str, default='both')
    parser.add_argument('--map_eval', type = ast.literal_eval, default=False)
    parser.add_argument('--num_batched_modules', type = int, default=0, help='Number of batches, not number of modules in '
                                                                             'each batch')
    parser.add_argument('--minimum_norm', type = ast.literal_eval, default=False)
    parser.add_argument('--module_names_path', type = str, default='')
    parser.add_argument('--include_last_layer', type = ast.literal_eval, default=False)
    parser.add_argument('--last_layer_full', type = ast.literal_eval, default=False)
    parser.add_argument('--percentile_range', type = str, default = "")
    parser.add_argument('--percentile_list', type=str, default="")
    args = parser.parse_args()

    args = parse_percentile_ramping_specification(args)
    # args = parse_percentile_list(args)
    # RTE experiment with percentile list
    keys = ['1.048888888888889', '1.767777777777778', '2.486666666666667', '3.2055555555555557', '3.9244444444444446', '4.6433333333333335', '5.362222222222222', '6.081111111111111', '6.8', '100', '12.6', '18.4', '24.2', '30.0', '33']
    keys = [float(key) for key in keys]
    args.percentile_list = keys
    if args.map_eval:
        if args.dataset_name == 'mrpc' or args.dataset_name == 'qqp' or args.dataset_name == 'qnli' or args.dataset_name == 'rte':
            run_map_eval(args, nli=True)
        else:
            run_map_eval(args)

    if args.experiment not in EXPERIMENTS:
        raise ValueError(f"Experiment {args.experiment} is not a valid experiment. Choose from {EXPERIMENTS}")

    if args.experiment == 'last_layer':
        sequential_last_layer(args)

    if args.experiment == 'random_ramping':
        run_random_ramping_experiments(args)

    if args.experiment == 'operator_norm_ramping':
        run_max_norm_ramping_experiments(args)

    if args.experiment == 'operator_norm_ramping_subclass':
        if args.subclass:
            run_max_norm_ramping_only_subclass(args)

    if args.experiment == 'sublayer_full':
        run_sublayer_ramping_full(args)

    if args.experiment == 'sublayer_predefined':
        run_sublayer_ramping_predefined_modules(args)

    ############################# NLI experiments ####################################

    if args.experiment == 'nli_last_layer':
        sequential_last_layer(args, nli=True)

    if args.experiment == 'nli_random_ramping':
        run_random_ramping_experiments(args, nli=True)

    if args.experiment == 'nli_operator_norm_ramping':
        run_max_norm_ramping_experiments(args, nli=True)

    if args.experiment == 'nli_sublayer_full':
        run_sublayer_ramping_full(args, nli=True)

    if args.experiment == 'nli_sublayer_predefined':
        run_sublayer_ramping_predefined_modules(args, nli=True)



