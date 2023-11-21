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
import uncertainty_toolbox as uct
from torch.nn import BCELoss, CrossEntropyLoss
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError

def save_laplace(save_path, laplace_cfg, **kwargs):
    pass


class Evaluator:
    def __init__(self, predictions, labels):

        self.predictions = predictions
        self.labels = labels

        self.multi_class = predictions.shape[-1] > 1
        self.nll_metric = CrossEntropyLoss() if self.multi_class else BCELoss
        self.calibration_class = MulticlassCalibrationError if self.multi_class else BinaryCalibrationError
        self.all_standard_metrics = {'f1 score': f1_score, 'balanced_accuracy_score': balanced_accuracy_score,
                                'accuracy_score': accuracy_score}


        self.torchmetrics_ = {'ECE': self.calibration_class(n_bins=20, norm='l1', num_classes=predictions.shape[-1]),
                              'MCE': self.calibration_class(n_bins=20, norm='max', num_classes=predictions.shape[-1]),
                               'RMSCE': self.calibration_class(n_bins=20, norm='l2', num_classes=predictions.shape[-1])}

        self.results = {}
    def compute_torch_metrics(self, results = None):

        if results is None:
            results = {}
        results['nll'] = self.nll_metric(self.predictions, self.labels)

        for key, val in self.torchmetrics_.items():
            results[key] = val(self.predictions, self.labels).item()

        return results

    def compute_standard_metrics(self,results = None, copy = True):

        if results is None:
            results = {}

        if copy:
            predictions, labels = self.predictions.clone().numpy(), self.labels.clone().numpy()
        else:
            predictions, labels = self.predictions.numpy(), self.labels.numpy()

        for key, val in self.all_standard_metrics.items():
            results[key] = val(labels, predictions.argmax(-1))

        return results

    def get_all_metrics(self, copy = False, override = True):

        results = self.compute_torch_metrics()
        results = self.compute_standard_metrics(results = results, copy = copy)
        if len(self.results) == 0 or override:
            self.results = results
        return results

def evaluate_laplace(la, trainer: Trainer, eval_dataset = None):

    eval_dataset = trainer.get_eval_dataloader(eval_dataset = eval_dataset)

    la.model.eval()
    predictions, labels = [], []
    for step, x in enumerate(eval_dataset):
        output = la(x)
        predictions.append(output)
        labels.append(x['labels'])

    predictions, labels = (torch.cat(predictions, dim = 0).detach().cpu(),
                           torch.cat(labels, dim = 0).detach().cpu())

    evaluator = Evaluator(predictions, labels)
    evaluator.get_all_metrics()
    return evaluator


def evaluate_swag(swag: PartialConstructorSwag,trainer: Trainer, eval_dataset = None):

    eval_dataset = trainer.get_eval_dataloader(eval_dataset=eval_dataset)
    swag.eval()

    predictions, labels = [], []
    for step, x in enumerate(eval_dataset):
        output = swag.predict_mc(**x)
        predictions.append(output)
        labels.append(x['labels'])

    predictions, labels = (torch.cat(predictions, dim = 0).detach().cpu(),
                           torch.cat(labels, dim = 0).detach().cpu())

    evaluator = Evaluator(predictions, labels)
    evaluator.get_all_metrics()
    return evaluator
