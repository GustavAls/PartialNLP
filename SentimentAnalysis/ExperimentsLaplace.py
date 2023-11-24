import ast
import copy
import os
import pickle

import torch
from datasets import load_dataset
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

class LaplaceExperiments:
    def __init__(self, args):
        self.default_args = {'output_path': args.output_path,
                             'train_batch_size': 1, 'eval_batch_size': 1, 'device': 'cpu', 'num_epochs': 1.0,
                             'dataset_name': 'imdb',
                             'train': True, 'train_size': 2, 'test_size': 2, 'device_batch_size': 1,
                             'learning_rate': 5e-05, 'seed': 0,'val_size': 2,
                             'laplace': True, 'swag': False, 'save_strategy': 'no',
                             'load_best_model_at_end': False, 'no_cuda': False}

        self.base_full_prior = 1e4
        self.base_min_prior = 1
        self.num_params = 0
        self.num_stoch_params = 0
        self.args = args
        self.module_names = None
        self.num_modules = [1, 2, 3, 4, 5, 8, 11, 17, 28, 38]

        default_args = Namespace(**self.default_args)
        self.default_args = default_args
        default_args.model_path = args.model_path
        self.sentiment_classifier = prepare_sentiment_classifier(default_args)

        self.minimum_prior, self.maximum_prior, self.best_nll = 1e-1, 1e5, np.inf
        self.train_loader, self.trainer, self.tokenized_val = self.sentiment_classifier.prepare_laplace(
            output_path=default_args.output_path,
            train_bs=default_args.train_batch_size,
            eval_bs=default_args.eval_batch_size,
            dataset_name=default_args.dataset_name,
            device_batch_size=default_args.device_batch_size,
            lr=default_args.learning_rate,
            data_path = args.data_path,
            run=args.run_number)

        self.subclass = None
        if not isinstance(self.sentiment_classifier.model, Extension):
            self.model = Extension(self.sentiment_classifier.model)
        else:
            self.model = self.sentiment_classifier.model

    def create_partial_random_ramping_construction(self, num_params):
        partial_constructor = self.use_subclass_part_only(PartialConstructor(self.model))
        partial_constructor.select_random_percentile(num_params)
        partial_constructor.select()
        self.num_stoch_params = partial_constructor.get_num_stochastic_parameters()
        self.num_params = partial_constructor.get_num_params()
        self.module_names = partial_constructor.module_names

    def create_partial_max_norm_ramping(self, num_params):
        partial_constructor = self.use_subclass_part_only(PartialConstructor(self.model))
        partial_constructor.select_max_operator_norm(num_params)
        partial_constructor.select()
        self.num_stoch_params = partial_constructor.get_num_stochastic_parameters()
        self.num_params = partial_constructor.get_num_params()
        self.module_names = partial_constructor.module_names

    def use_subclass_part_only(self, partial_constructor):

        if self.subclass == 'attn':
            partial_constructor.set_use_only_attn()
        elif self.subclass == 'mlp':
            partial_constructor.set_use_only_mlp()

        return partial_constructor

    def fit_laplace(self, prior_precision=1.0):
        la = lp.Laplace(self.model, 'classification',
                        subset_of_weights='all',  # Deprecated
                        hessian_structure='kron',
                        prior_precision=prior_precision)

        la.fit(self.train_loader)

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

    def optimize_prior_precision(self, num_steps=7, use_uninformed = False):

        self.best_nll = np.inf
        negative_log_likelihoods = []
        priors = [self.minimum_prior, self.maximum_prior]
        if use_uninformed:
            UserWarning("Optimizing called but using uninformed prior")
            percentage_of_stoch_params = self.num_stoch_params/self.num_params
            coeff = (self.base_full_prior - self.base_min_prior)
            prior = self.base_min_prior + coeff * percentage_of_stoch_params
            la = self.fit_laplace(prior_precision=prior)
            return la

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

    def random_ramping_experiment(self, run_number = 0, use_uninformed = False):

        print("Running random ramping experiment on ", self.default_args.dataset_name)
        results = {'results': {}, 'module_selection': {}}

        save_path = os.path.join(self.args.output_path, 'random_module_ramping')
        self.ensure_path_existence(save_path)
        for num_modules in self.num_modules:
            self.create_partial_random_ramping_construction(num_modules)
            la = self.optimize_prior_precision(self.args.num_optim_steps, use_uninformed = use_uninformed)
            evaluator = utils.evaluate_laplace(la, self.trainer)
            evaluator.results['prior_precision'] = self.best_nll
            results['results'][num_modules] = copy.deepcopy(evaluator)
            results['module_selection'][num_modules] = copy.deepcopy(self.module_names)

            with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def max_norm_ramping_experiment(self, run_number = 0, use_uninformed = False):

        print("Running max norm ramping experiment on ", self.default_args.dataset_name)
        results = {'results': {}, 'module_selection': {}}

        for num_modules in self.num_modules:
            self.create_partial_max_norm_ramping(num_modules)
            la = self.optimize_prior_precision(self.args.num_optim_steps, use_uninformed = use_uninformed)
            evaluator = utils.evaluate_laplace(la, self.trainer)
            evaluator.results['prior_precision'] = self.best_nll
            results['results'][num_modules] = copy.deepcopy(evaluator)
            results['module_selection'][num_modules] = copy.deepcopy(self.module_names)

        save_path = os.path.join(self.args.output_path, 'operator_norm_module_ramping')
        self.ensure_path_existence(save_path)
        with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_random_ramping_experiments(args):

    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    la_args = {'model_path': model_path,
               'dataset_name': args.dataset_name,
               'num_optim_steps': 7,
               'data_path': data_path,
               'run_number': args.run_number,
               'output_path': args.output_path}

    # la_args['model_path']= r"C:\Users\45292\Documents\Master\SentimentClassification\checkpoint-782"
    la_args = Namespace(**la_args)
    lap_exp = LaplaceExperiments(args = la_args)
    lap_exp.random_ramping_experiment(args.run_number, args.uninformed_prior)


def run_max_norm_ramping_experiments(args):

    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    la_args = {'model_path': model_path,
               'dataset_name': args.dataset_name,
               'num_optim_steps': 7,
               'data_path': data_path,
               'run_number': args.run_number,
               'output_path': args.output_path}

    # la_args['model_path']= r"C:\Users\45292\Documents\Master\SentimentClassification\checkpoint-782"
    la_args = Namespace(**la_args)
    lap_exp = LaplaceExperiments(args = la_args)
    lap_exp.max_norm_ramping_experiment(args.run_number, args.uninformed_prior)

def run_max_norm_ramping_only_subclass(args):
    data_path = args.data_path
    model_ext_path = [path for path in os.listdir(data_path) if 'checkpoint' in path][0]

    model_path = os.path.join(data_path, model_ext_path)
    args.model_path = model_path
    la_args = {'model_path': model_path,
               'dataset_name': args.dataset_name,
               'num_optim_steps': 7,
               'data_path': data_path,
               'run_number': args.run_number,
               'output_path': args.output_path}

    # la_args['model_path']= r"C:\Users\45292\Documents\Master\SentimentClassification\checkpoint-782"
    la_args = Namespace(**la_args)
    lap_exp = LaplaceExperiments(args = la_args)
    lap_exp.number_of_modules = [] # TODO fix correct here
    lap_exp.subclass = args.subclass
    lap_exp.max_norm_ramping_experiment(args.run_number, args.uninformed_prior)



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
    parser.add_argument('--subclass', type = str, default='')
    args = parser.parse_args()

    if args.experiment == 'random_ramping':
        run_random_ramping_experiments(args)
    if args.experiment == 'operator_norm_ramping':
        run_max_norm_ramping_experiments(args)
    if args.experiment == 'operator_norm_ramping_subclass':
        if args.subclass:
            run_max_norm_ramping_only_subclass(args)


