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


class SentimentClassifier:
    def __init__(self, network_name, id2label, label2id, train_size=300, test_size=30):
        self._tokenizer = AutoTokenizer.from_pretrained(network_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(network_name,
                                                                        num_labels=2,
                                                                        id2label=id2label,
                                                                        label2id=label2id)

        self.model.save_pretrained(os.path.join(os.getcwd(), "model"))
        self.collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        self.train_size = train_size
        self.test_size = test_size

    def load_text_dataset(self, dataset_name="imdb"):
        data = load_dataset(dataset_name)
        train_data = data["train"].shuffle(seed=42).select([i for i in list(range(self.train_size))])
        test_data = data["test"].shuffle(seed=42).select([i for i in list(range(self.test_size))])
        del data
        return train_data, test_data

    def tokenize(self, examples):
        return self._tokenizer(examples["text"], truncation=True)

    @staticmethod
    def compute_metrics(eval_pred):
        load_accuracy = evaluate.load("accuracy")
        load_f1 = evaluate.load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    def runner(self, output_path, train_bs, eval_bs, num_epochs, dataset_name, device_batch_size, train=True):
        train_data, test_data = self.load_text_dataset(dataset_name=dataset_name)
        tokenized_train = train_data.map(self.tokenize, batched=True, batch_size=train_bs)
        tokenized_test = test_data.map(self.tokenize, batched=True, batch_size=eval_bs)

        training_args = TrainingArguments(output_dir=output_path,
                                          learning_rate=2e-5,
                                          do_train=train,
                                          optim='sgd',
                                          per_device_train_batch_size=device_batch_size,
                                          per_device_eval_batch_size=device_batch_size,
                                          num_train_epochs=num_epochs,
                                          evaluation_strategy="epoch",
                                          save_strategy="epoch",
                                          load_best_model_at_end=True,
                                          weight_decay=0.01)

        self.model = PartialConstructorSwag(self.model, n_iterations_between_snapshots=1,
                                            module_names=['classifier'],
                                            num_columns=10)
        self.model.select()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self._tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics
        )

        if train:
            trainer.train()
            print("Training is done")
        else:
            trainer.evaluate()
            print("Evaluation is done")

    def prepare_laplace(self, output_path, train_bs, eval_bs, dataset_name, device_batch_size):
        """

        :param output_path: Ouput path, for compatibility with other function calls, this function does not save
        :param train_bs: training batch size
        :param eval_bs: eval batch size
        :param dataset_name: (str) name of the dataset
        :param device_batch_size: per device batch size
        :return: None
        """
        train_data, test_data = self.load_text_dataset(dataset_name=dataset_name)
        tokenized_train = train_data.map(self.tokenize, batched=True, batch_size=train_bs)
        tokenized_test = test_data.map(self.tokenize, batched=True, batch_size=eval_bs)

        training_args = TrainingArguments(output_dir=output_path,
                                          learning_rate=2e-5,
                                          do_train=True,
                                          per_device_train_batch_size=device_batch_size,
                                          per_device_eval_batch_size=device_batch_size,
                                          num_train_epochs=1,
                                          evaluation_strategy="epoch",
                                          save_strategy="epoch",
                                          load_best_model_at_end=True,
                                          weight_decay=0.01,
                                          laplace=True
                                          )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self._tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics
        )

        epoch_iterator, model, trainer = trainer.train()
        self.model = model
        return epoch_iterator, trainer


def prepare_sentiment_classifier(args, model_name="distilbert-base-uncased"):
    if args.dataset_name.lower() == 'imdb':
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    else:
        raise NotImplementedError("Only implemented so far for dataset_name == 'imdb'")

    sentiment_classifier = SentimentClassifier(model_name,
                                               id2label=id2label,
                                               label2id=label2id,
                                               train_size=args.train_size,
                                               test_size=args.test_size)

    return sentiment_classifier


def prepare_and_run_sentiment_classifier(args, sentiment_classifier=None):
    if sentiment_classifier is None:
        sentiment_classifier = prepare_sentiment_classifier(args)

    sentiment_classifier.runner(output_path=args.output_path,
                                train_bs=args.train_batch_size,
                                eval_bs=args.eval_batch_size,
                                num_epochs=args.num_epochs,
                                dataset_name=args.dataset_name,
                                device_batch_size=args.device_batch_size,
                                train=args.train)

    return None


def construct_laplace(sent_class, laplace_cfg, args):
    train_loader, trainer = sent_class.prepare_laplace(output_path=args.output_path,
                                                       train_bs=args.train_batch_size,
                                                       eval_bs=args.eval_batch_size,
                                                       dataset_name=args.dataset_name,
                                                       device_batch_size=args.device_batch_size,
                                                       )

    if not isinstance(sent_class.model, Extension):
        model = Extension(sent_class.model)
    else:
        model = sent_class.model

    for idx, module_name in enumerate(laplace_cfg.module_names):
        if module_name.split(".")[0] != 'model':
            laplace_cfg.module_names[idx] = ".".join(('model', module_name))

    partial_constructor = PartialConstructor(model, module_names=laplace_cfg.module_names)
    partial_constructor.select()

    la = lp.Laplace(model, laplace_cfg.ml_task,
                    subset_of_weights=laplace_cfg.deprecated_subset_of_weights,
                    hessian_structure=laplace_cfg.hessian_structure,
                    prior_precision=laplace_cfg.prior_precision,
                    subnetwork_indices=partial_constructor.get_subnetwork_indices())

    la.fit(train_loader)

    return la


def construct_swag(sentiment_classifier, swag_cfg):
    sentiment_classifier.model = PartialConstructorSwag(
        sentiment_classifier.model,
        n_iterations_between_snapshots=swag_cfg.n_iterations_between_snapshots,
        module_names=swag_cfg.module_names,
        num_columns=swag_cfg.number_of_columns_in_D,
        num_mc_samples=swag_cfg.num_mc_samples,
        min_var=swag_cfg.min_var,
        reduction=swag_cfg.reduction,
        num_classes=swag_cfg.num_classes
    )

    return sentiment_classifier


def save_object_(obj, output_path, swag_or_laplace='laplace', cfg=None):

    changed_things = check_cfg_status(cfg, swag_or_laplace)

    if swag_or_laplace == 'laplace':
        to_be_saved = {'la': obj, 'cfg': cfg}
    elif swag_or_laplace == 'swag':
        to_be_saved = {'swag': obj, 'cfg': cfg}
    else:
        raise ValueError("swag_or_laplace should be a string in ['swag', 'laplace']")


    if len(changed_things) == 0:
        filename = swag_or_laplace + "_" + 'default.pkl'
    else:
        filename = "_".join([swag_or_laplace] + [it for item in changed_things.items() for it in item])

    if os.path.isdir(output_path):

        save_name = os.path.join(
            output_path, ".".join((filename, "pkl"))
        )
    elif output_path.split(".")[-1] in ['pkl', 'pt']:
        save_name = output_path
    else:
        raise ValueError("output path should either specify dir or be a full path with either .pkl or .pt ending")

    if ".pt" in save_name:
        torch.save(to_be_saved, save_name)
    else:
        with open(save_name, 'wb') as handle:
            pickle.dump(to_be_saved, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_cfg_status(cfg, swag_or_laplace = 'laplace'):

    if cfg is None:
        return {}
    elif isinstance(cfg, str):
        try:
            cfg_dict = yaml.safe_load(open(cfg, 'r'))
        except:
            return {}
    else:
        cfg_dict = cfg.__dict__

    default_cfg = yaml.safe_load(open("_".join((swag_or_laplace, 'cfg', 'default.yaml')), 'r'))

    changed_things = {}
    for key, val in default_cfg.items():
        cfg_vals = cfg_dict[key]

        if val == cfg_vals:
            continue

        if key == 'module_names':
            if all(('model.' in module_name for module_name in cfg_vals)):
                if [module_name.replace("model.") for module_name in cfg_vals] == val:
                    continue

            changed_things[key] = f'num_of_{len(cfg_vals)}'
            continue

        changed_things[key] = str(cfg_vals)

    return changed_things


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
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--train", type=ast.literal_eval, default=True)
    parser.add_argument("--train_size", type=int, default=24)
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--device_batch_size", type=int, default=12)
    parser.add_argument('--laplace', type=ast.literal_eval, default=False)
    parser.add_argument('--swag', type=ast.literal_eval, default=False)
    parser.add_argument('--swag_cfg', default=None)
    parser.add_argument('--la_cfg', default=None)
    args = parser.parse_args()

    if not any((args.swag, args.laplace)):
        prepare_and_run_sentiment_classifier(args)

    elif args.swag:
        cfg = Namespace(**yaml.safe_load(open(
            'swag_cfg.yaml' if args.swag_cfg is None else args.swag_cfg, 'r')
        ))
        if not args.train:
            raise NotImplementedError("Swag model not yet implemented with evaluation protocol")

        sentiment_classifier = prepare_sentiment_classifier(args)
        sentiment_classifier = construct_swag(sentiment_classifier, cfg)
        prepare_and_run_sentiment_classifier(args, sentiment_classifier)
        save_object_(sentiment_classifier.model, args.output_path, 'swag', args.swag_cfg)

    elif args.laplace:
        cfg = Namespace(**yaml.safe_load(open(
            'laplace_cfg.yaml' if args.la_cfg is None else args.la_cfg, 'r')
        ))
        sentiment_classifier = prepare_sentiment_classifier(args)
        la = construct_laplace(sentiment_classifier, cfg, args)

        save_object_(la, args.output_path, 'laplace', args.la_cfg)
