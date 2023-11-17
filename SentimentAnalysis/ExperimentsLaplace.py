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


class LaplaceExperiments:
    def __init__(self, args):
        self.default_args = {'output_path': 'C:\\Users\\45292\\Documents\\Master\\SentimentClassification',
                             'train_batch_size': 32, 'eval_batch_size': 32, 'device': 'cpu', 'num_epochs': 1.0,
                             'dataset_name': 'imdb',
                             'train': True, 'train_size': None, 'test_size': None, 'device_batch_size': 32,
                             'learning_rate': 5e-05, 'seed': 0,
                             'laplace': True, 'swag': False, 'save_strategy': 'no',
                             'load_best_model_at_end': False, 'no_cuda': False}

        self.args = args
        self.num_modules = [1, 2, 3, 4, 5, 8, 11, 17, 28, 38]

        default_args = Namespace(**self.default_args)
        default_args.model_path = args.model_path
        self.sentiment_classifier = prepare_sentiment_classifier(default_args)

        self.minimum_prior, self.maximum_prior, self.best_nll = 1.0, 1e5, np.inf
        self.train_loader, self.trainer, self.tokenized_val = self.sentiment_classifier.prepare_laplace(
            output_path=default_args.output_path,
            train_bs=default_args.train_batch_size,
            eval_bs=default_args.eval_batch_size,
            dataset_name=default_args.dataset_name,
            device_batch_size=default_args.device_batch_size,
            lr=default_args.learning_rate)

        if not isinstance(self.sentiment_classifier.model, Extension):
            self.model = Extension(self.sentiment_classifier.model)
        else:
            self.model = self.sentiment_classifier.model

    def create_partial_random_ramping_construction(self, num_params):
        partial_constructor = PartialConstructor(self.model)
        partial_constructor.select_random_percentile(num_params)
        partial_constructor.select()

    def fit_laplace(self, prior_precision=1.0):
        la = lp.Laplace(self.model, 'classification',
                        subset_of_weights='all',  # Deprecated
                        hessian_structure='kron',
                        prior_precision=prior_precision)

        la.fit(self.train_loader)

        return la

    def train_and_predict_new_prior(self, priors, nlls):

        prediction_space = np.linspace(self.minimum_prior, self.maximum_prior, 1000)
        gaussian_process = GaussianProcessRegressor().fit(priors, nlls)
        predictions, std = gaussian_process.predict(prediction_space, return_std=True)
        minimum_prediction = prediction_space[np.argmin(predictions - std)]
        return minimum_prediction

    def optimize_prior_precision(self, num_steps=7):

        negative_log_likelihoods = []
        priors = [self.minimum_prior, self.maximum_prior]
        la = self.fit_laplace(prior_precision=self.minimum_prior)
        best_la = copy.deepcopy(la)
        evaluator = utils.evaluate_laplace(la, self.trainer, self.tokenized_val)
        negative_log_likelihoods.append(evaluator['nll'])
        self.best_nll = negative_log_likelihoods[-1]

        la = self.fit_laplace(prior_precision=self.maximum_prior)
        evaluator = utils.evaluate_laplace(la, self.trainer, self.tokenized_val)
        negative_log_likelihoods.append(evaluator['nll'])

        if negative_log_likelihoods[-1] < self.best_nll:
            best_la = copy.deepcopy(la)
            self.best_nll = negative_log_likelihoods[-1]

        for step in range(num_steps):
            next_point = self.train_and_predict_new_prior(priors, negative_log_likelihoods)
            priors.append(next_point)

            la = self.fit_laplace(prior_precision=self.maximum_prior)
            evaluator = utils.evaluate_laplace(la, self.trainer, self.tokenized_val)
            negative_log_likelihoods.append(evaluator['nll'])
            if negative_log_likelihoods < self.best_nll:
                best_la = copy.deepcopy(la)
                self.best_nll = negative_log_likelihoods[-1]

        return best_la

    def ensure_path_existence(self, path):

        if not os.path.exists(path):
            if not os.path.exists(os.path.dirname(path)):
                raise ValueError("Make at least the folder over where the experiments are being put")
            else:
                os.mkdir(path)

    def random_ramping_experiment(self, run_number = 0):

        print("Running random ramping experiment on ", self.default_args.dataset_name)
        results = {}

        for num_modules in self.num_modules:
            self.create_partial_random_ramping_construction(num_modules)
            la = self.optimize_prior_precision(self.args.num_optim_steps)
            evaluator = utils.evaluate_laplace(la, self.trainer)
            evaluator.results['prior_precision'] = self.best_nll
            results[num_modules] = copy.deepcopy(evaluator)

        save_path = os.path.join(self.args.output_path, 'random_module_ramping')
        self.ensure_path_existence(save_path)
        with open(os.path.join(save_path, f'run_number_{run_number}.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


