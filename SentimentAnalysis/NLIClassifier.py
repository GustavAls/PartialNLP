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
        super().__init__(network_name, id2label,label2id,train_size,val_size,test_size,dataset_name)

    def load_text_dataset(self, dataset_name='rte'):
        dataset = load_dataset("glue", dataset_name)
        seed = np.random.randint(0,100)

        dataset['test'] = dataset.pop('validation')
        dataset['train'] = dataset['train'].train_test_split(test_size = 0.1, shuffle = True, seed=seed)

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
        return self._tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

    def to_dataframe(self, dataset):
        dataframe = pd.DataFrame()
        dataframe['sentence1'] = dataset['sentence1']
        dataframe['sentence2'] = dataset['sentence2']
        dataframe['label'] = dataset['label']
        dataframe['idx'] = dataset['idx']
        return dataframe






