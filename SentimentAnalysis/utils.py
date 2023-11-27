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
    def __init__(self, predictions, labels, has_seen_softmax = False):

        self.predictions = predictions
        self.labels = labels

        self.multi_class = predictions.shape[-1] > 1
        self.nll_metric = CrossEntropyLoss() if self.multi_class else BCELoss
        if has_seen_softmax:
            self.nll_metric = evaluate_loss
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
        results['nll'] = self.nll_metric(self.predictions, self.labels).item()

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

    evaluator = Evaluator(predictions, labels, has_seen_softmax=True)
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

    evaluator = Evaluator(predictions, labels, has_seen_softmax=True)
    evaluator.get_all_metrics()
    return evaluator


def evaluate_map(model, trainer: Trainer, eval_dataset = None):
    eval_dataset = trainer.get_eval_dataloader(eval_dataset=eval_dataset)
    predictions, labels = [], []
    model.eval()
    with torch.no_grad():
        for step, x in enumerate(eval_dataset):
            output = model(**x).detach().cpu()
            predictions.append(output)
            labels.append(x['labels'].detach().cpu())

    predictions, labels = (torch.cat(predictions, dim=0).detach().cpu(),
                           torch.cat(labels, dim=0).detach().cpu())

    evaluator = Evaluator(predictions, labels)
    evaluator.get_all_metrics()
    return evaluator

def evaluate_loss(predictions, labels, use_softmax = False):

    loss = []
    predictions = predictions.clone().numpy()
    labels = labels.clone().numpy()
    labels = to_one_hot(predictions, labels)
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
    loss_fn = lambda pred, lab: -np.sum(np.log(pred) * lab)
    for pred, lab in zip(predictions, labels):
        if use_softmax:
            pred = softmax(pred)
        loss.append(loss_fn(pred, lab))

    return np.mean(loss)

def to_one_hot(predictions, labels):
    labels_new = np.zeros_like(predictions)
    labels_new[np.arange(len(labels_new)), labels] = 1
    return labels_new


def run_evaluator_again(predictions, labels):

    m = nn.Softmax(dim = 1)
    predictions = m(predictions)
    evaluator = Evaluator(predictions, labels)
    evaluator.get_all_metrics()
    breakpoint()

def evaluate_loss_with_only_wrong(predictions, labels):

    where = np.argmax(predictions, -1) == labels
    return predictions[~where], labels[~where]
if __name__ == '__main__':

    laplace_path = r"C:\Users\45292\Documents\Master\SentimentClassification\Laplace\run_0\run_number_0.pkl"
    map_path = r"C:\Users\45292\Documents\run_0\run_number_0_map.pkl"

    lap = pickle.load(open(laplace_path, 'rb'))
    map = pickle.load(open(map_path, 'rb'))
    # run_evaluator_again(predictions=map['results'].predictions,labels = map['results'].labels)
    print(evaluate_loss(map['results'].predictions, map['results'].labels, use_softmax=True))
    eval_ = Evaluator(map['results'].predictions, map['results'].labels, has_seen_softmax=False)
    results = {0 : eval_.get_all_metrics()}
    for key in lap['results'].keys():
        eval_ = Evaluator(lap['results'][key].predictions, lap['results'][key].labels, has_seen_softmax=True)
        results[key] = eval_.get_all_metrics()
    breakpoint()
