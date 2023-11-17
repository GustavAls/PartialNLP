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



def bisect_laplace(sent_class, laplace_cfg, args):
    train_loader, trainer = sent_class.prepare_laplace(output_path=args.output_path,
                                                       train_bs=args.train_batch_size,
                                                       eval_bs=args.eval_batch_size,
                                                       dataset_name=args.dataset_name,
                                                       device_batch_size=args.device_batch_size,
                                                       lr=args.learning_rate)

    if not isinstance(sent_class.model, Extension):
        model = Extension(sent_class.model)
    else:
        model = sent_class.model

    for idx, module_name in enumerate(laplace_cfg.module_names):
        if module_name.split(".")[0] != 'model':
            laplace_cfg.module_names[idx] = ".".join(('model', module_name))

    partial_constructor = PartialConstructor(model, module_names='model.classifier')
    partial_constructor.select()

    min_prior, max_prior = laplace_cfg.prior_precision


def construct_and_train_la(model, laplace_cfg, train_loader):
    la = lp.Laplace(model, laplace_cfg.ml_task,
                    subset_of_weights=laplace_cfg.deprecated_subset_of_weights,
                    hessian_structure=laplace_cfg.hessian_structure,
                    prior_precision=laplace_cfg.prior_precision)

    la.fit(train_loader)
    return la


def get_bisect_results(model, laplace_cfg, train_loader, eval_loader, prec):
    laplace_cfg.prior_precision = prec
    la = construct_and_train_la(model, laplace_cfg, train_loader)
    res = evaluate_laplace(la, eval_loader)
    return res
def bisect_la(model, laplace_cfg, train_loader, eval_loader, min_prec, max_prec, counter = 0):

    precision_results = {}
    precision_results[min_prec] = get_bisect_results(model, laplace_cfg, train_loader, eval_loader, min_prec)
    precision_results[max_prec] = get_bisect_results(model, laplace_cfg, train_loader, eval_loader, max_prec)
    while counter <= 5:
        middle_prec = (min_prec + max_prec)/2
        precision_results[middle_prec]  = get_bisect_results(
            model, laplace_cfg, train_loader, eval_loader, middle_prec)

        tried_precisions = sorted(list(precision_results.keys()))
        results = [precision_results[key]['nll'] for key in tried_precisions]
        minimum_value = np.argmin(results)

        if minimum_value == 0:
            max_prec = tried_precisions[1]
            min_prec = tried_precisions[0]
        elif minimum_value == len(precision_results)-1:
            max_prec = tried_precisions[-1]
            min_prec = tried_precisions[-2]
        else:
            if results[minimum_value + 1] <= results[minimum_value -1]:
                min_prec = tried_precisions[minimum_value]
                max_prec = tried_precisions[minimum_value +1]
            else:
                min_prec = tried_precisions[minimum_value -1]
                max_prec = tried_precisions[minimum_value + 1]


    precisions = sorted(list(precision_results.keys()))
    results = [precision_results[prec] for prec in precisions]
    return results[np.argmin(results)]


def evaluate_laplace(la, eval_loader):
    return None