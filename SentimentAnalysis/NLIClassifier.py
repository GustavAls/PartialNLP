import ast
import copy
import os
import pickle
import torch
from datasets import load_dataset
from datasets import Dataset as HuggingFaceDataset
import pandas as pd
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
from SentimentClassifier import SentimentClassifier
import random


class NLIClassifier(SentimentClassifier):

    def __init__(self, network_name,
                 id2label=None,
                 label2id=None,
                 train_size=None,
                 val_size=None,
                 test_size=None,
                 dataset_name="rte"):
        super().__init__(network_name, id2label, label2id, train_size, val_size, test_size, dataset_name)
        self.dataset_cols = {"mrpc": ["sentence1", "sentence2", "label"],
                             "qqp": ["question1", "question2", "label"],
                             "qnli": ["question", "sentence", "label"],
                             "rte": ["sentence1", "sentence2", "label"]}

    def load_text_dataset(self, dataset_name='rte'):
        dataset = load_dataset("glue", dataset_name)
        seed = np.random.randint(0, 100)

        if 0 < self.train_size < 1:
            self.train_size = int(self.train_size * len(dataset['train']))

        dataset['test'] = dataset.pop('validation')
        dataset['train'] = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=seed)

        if self.train_size != 1 and isinstance(self.train_size, int):
            train_data = dataset['train']['train'].shuffle(seed=42).select([i for i in range(self.train_size)])
        else:
            train_data = dataset['train']['train']
        if self.test_size != 1 and isinstance(self.test_size, int):
            test_data = dataset['test'].shuffle(seed=42).select([i for i in range(self.train_size)])
        else:
            test_data = dataset['test']

        if self.val_size != 1 and isinstance(self.val_size, int):
            val_data = dataset['train']['test'].shuffle(seed=42).select([i for i in range(self.train_size)])
        else:
            val_data = dataset['train']['test']

        return train_data, test_data, val_data

    def tokenize(self, examples):
        return self._tokenizer(examples[self.dataset_cols[self.dataset_name][0]],
                               examples[self.dataset_cols[self.dataset_name][1]],
                               truncation=True, padding="max_length")

    def to_dataframe(self, dataset):
        dataframe = pd.DataFrame()
        dataframe[self.dataset_cols[self.dataset_name][0]] = dataset[self.dataset_cols[self.dataset_name][0]]
        dataframe[self.dataset_cols[self.dataset_name][1]] = dataset[self.dataset_cols[self.dataset_name][1]]
        dataframe[self.dataset_cols[self.dataset_name][2]] = dataset[self.dataset_cols[self.dataset_name][2]]
        try:
            dataframe['idx'] = dataset['idx']
        except:
            pass
        return dataframe


def prepare_nli_classifier(args, model_name='distilbert-base-uncased', train_size=1):
    if train_size >= 1:
        train_size = args.train_size

    if hasattr(args, 'model_path'):
        model_path_or_name = args.model_path
        NLI = NLIClassifier(network_name=model_path_or_name,
                            train_size=train_size,
                            val_size=args.val_size,
                            test_size=args.test_size,
                            dataset_name=args.dataset_name)
    else:
        NLI = NLIClassifier(network_name=model_name,
                            train_size=train_size,
                            val_size=args.val_size,
                            test_size=args.test_size,
                            dataset_name=args.dataset_name)
    return NLI


def run_datagen(args, network_name='distilbert-base-uncased'):
    nli_classifier = prepare_nli_classifier(args, network_name)
    nli_classifier.runner(output_path=args.output_path,
                          data_path=args.data_path,
                          train_bs=args.train_batch_size,
                          eval_bs=args.eval_batch_size,
                          num_epochs=args.num_epochs,
                          dataset_name=args.dataset_name,
                          device_batch_size=args.device_batch_size,
                          lr=args.learning_rate,
                          logging_perc=args.logging_perc,
                          save_strategy=args.save_strategy,
                          evaluation_strategy=args.evaluation_strategy,
                          metric_for_best_model=args.metric_for_best_model,
                          load_best_model_at_end=True,
                          no_cuda=args.no_cuda,
                          eval_steps=args.eval_steps,
                          run=args.run)


def dataramping(args, network_name='distilbert-base-uncased'):
    epochs = np.linspace(1, 10, 10, endpoint=True)
    train_sizes = [1.0 / num_epochs for num_epochs in epochs]

    for train_size, epoch in zip(train_sizes, epochs):
        args.num_epochs = epoch
        nli_classifier = prepare_nli_classifier(args, network_name, train_size=train_size)
        nli_classifier.runner(output_path=args.output_path,
                              data_path=args.data_path,
                              train_bs=args.train_batch_size,
                              eval_bs=args.eval_batch_size,
                              num_epochs=args.num_epochs,
                              dataset_name=args.dataset_name,
                              device_batch_size=args.device_batch_size,
                              lr=args.learning_rate,
                              logging_perc=args.logging_perc,
                              save_strategy=args.save_strategy,
                              evaluation_strategy=args.evaluation_strategy,
                              metric_for_best_model=args.metric_for_best_model,
                              load_best_model_at_end=True,
                              no_cuda=args.no_cuda,
                              eval_steps=args.eval_steps,
                              run=args.run)


if __name__ == "__main__":
    """

    Notes: We can avoid compatibility problems if we exempt embeddings and certain types of layer norm from 
    the gradient calculations. CHECK if this does not affect gradient computations on other modules. 

    If this is to be implemented, similar changes has to be made in the subnetwork choice/selection
    to avoid indexing errors, or just calculation mistakes without errors. 


    Changes in source code:
    HUGGINGFACE:
        training_args.py: Added Laplace input as optional, on line 715 (bool)
        trainer.py: Added function ._prepare_inner_training_for_laplace().
        trainer.py: Made a check for (args.laplace_partial == True) in .train() so it returns
                    ._prepare_inner_training_for_laplace(). 

    LAPLACE:
        subnetlaplace.py: Made new class to accomadate NLP 
        backpack.py: 3 new classes to accomodate NLP 
        subnetmask: added new ModuleNameSubnetMaskNLP class 
        utils.init: added ModuleNameSubnetMaskNLP to imports

    torch backpack\_init.py: change to skip embedding layer for backprop extension


    """

    parser = argparse.ArgumentParser(
        description="Run training and or evaluation of Sentiment Classifier"
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_epochs", type=float, default=1)
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--train_size", type=int, default=1)  # 1 gives full dataset
    parser.add_argument("--val_size", type=int, default=1)  # 1 gives full dataset
    parser.add_argument("--test_size", type=int, default=1)  # 1 gives full dataset
    parser.add_argument("--device_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-05)
    parser.add_argument('--swag_cfg', default=None)
    parser.add_argument('--la_cfg', default=None)
    parser.add_argument('--logging_perc', type=float, default=-1)  # -1 for default from transformers
    parser.add_argument('--eval_steps', type=float, default=-1)  # -1 for default from transformers
    parser.add_argument('--save_strategy', default='epoch')  # 'epoch' for default from transformers
    parser.add_argument('--evaluation_strategy', default='epoch')
    parser.add_argument('--no_cuda', type=ast.literal_eval, default=False)
    parser.add_argument('--dataramping', type=ast.literal_eval, default=False)
    parser.add_argument('--load_best_model_at_end', type=ast.literal_eval, default=False)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--metric_for_best_model', type=str, default='loss')
    parser.add_argument('--run', type=int, default=0)

    args = parser.parse_args()

    network_name = 'distilbert-base-uncased'

    if args.dataramping:
        dataramping(args, network_name)
    else:
        run_datagen(args, network_name)
