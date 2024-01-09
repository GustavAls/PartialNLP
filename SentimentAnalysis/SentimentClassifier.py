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


class SentimentClassifier:
    def __init__(self, network_name, id2label=None, label2id=None, train_size=None, val_size=None, test_size=None, dataset_name="sst2"):
        self._tokenizer = AutoTokenizer.from_pretrained(network_name)
        if id2label is None and label2id is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(network_name,
                                                                            num_labels=2)
        else:

            self.model = AutoModelForSequenceClassification.from_pretrained(network_name,
                                                                            num_labels=2,
                                                                            label2id=label2id,
                                                                            id2label=id2label)
        self.model.save_pretrained(os.path.join(os.getcwd(), "model"))
        self.collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.dataset_name = dataset_name
        self.column_name = { "sst2": "sentence",
                           "imdb": "text" }

    def load_text_dataset(self, dataset_name="imdb"):
        data = load_dataset(dataset_name)
        if 0 < self.train_size < 1:
            self.train_size = int(len(data['train']) * self.train_size)

        if self.train_size == 1.0:
            val_frac = 1/10 if dataset_name == "sst2" else 1/5
            train_data_length = len(data['train']['label'])
            validation_data_indices = set(list(np.random.choice(range(train_data_length), int(val_frac * train_data_length))))
            training_data_indices = set(range(train_data_length))-validation_data_indices
            train_data = data['train'].select(list(training_data_indices))
            val_data = data['train'].select(list(validation_data_indices))

            if dataset_name == "imdb":
                test_data = data["test"] if self.test_size == 1 else \
                    data["test"].shuffle(seed=42).select([i for i in list(range(int(self.test_size)))])

            elif dataset_name == "sst2":
                test_data = data["validation"] if self.test_size == 1 else \
                    data["validation"].shuffle(seed=42).select([i for i in list(range(int(self.test_size)))])

        else:
            train_data = data["train"].shuffle(seed=42).select([i for i in list(range(int(self.train_size)))])
            val_data = data["train"].shuffle(seed=42).select([i for i in list(range(int(self.val_size)))])

            if dataset_name == "imdb":
                test_data = data["test"] if self.test_size == 1 else \
                    data["test"].shuffle(seed=42).select([i for i in list(range(int(self.test_size)))])

            elif dataset_name == "sst2":
                test_data = data["validation"] if self.test_size == 1 else \
                    data["validation"].shuffle(seed=42).select([i for i in list(range(int(self.test_size)))])
            print("Training class split: " + str(sum(train_data['label'])/len(train_data['label'])))
            print("Test class split: " + str(sum(test_data['label'])/len(test_data['label'])))
        del data
        return train_data, test_data, val_data

    def tokenize(self, examples):
        return self._tokenizer(examples[self.column_name[self.dataset_name]], truncation=True)

    @staticmethod
    def compute_metrics(eval_pred):
        load_accuracy = evaluate.load("accuracy")
        load_f1 = evaluate.load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    def to_dataframe(self, dataset):
        dataframe = pd.DataFrame()
        dataframe[self.column_name[self.dataset_name]] = dataset[self.column_name[self.dataset_name]]
        dataframe['label'] = dataset['label']
        return dataframe

    def data_to_csv(self, train, val, test, output_path):
        dataframe_train = self.to_dataframe(train)
        dataframe_train.to_csv(os.path.join(output_path, f'train_data.csv'))
        dataframe_val = self.to_dataframe(val)
        dataframe_val.to_csv(os.path.join(output_path, f'val_data.csv'))
        dataframe_test = self.to_dataframe(test)
        dataframe_test.to_csv(os.path.join(output_path, f'test_data.csv'))
        print("Saved train, val, test to csv files in ")

    def load_save_dataset(self, data_path, dataset_name, run, output_path = None):
        split_names = ["train", "val", "test"]
        if data_path is None:
            train_data, test_data, val_data = self.load_text_dataset(dataset_name=dataset_name)
            self.data_to_csv(train_data, val_data, test_data, output_path)
        else:
            for split in split_names:
                data_csv_path = os.path.join(data_path, f'{split}_data.csv')

                if split == "train":
                    train_data = HuggingFaceDataset.from_csv(data_csv_path)
                elif split == "val":
                    val_data = HuggingFaceDataset.from_csv(data_csv_path)
                elif split == "test":
                    test_data = HuggingFaceDataset.from_csv(data_csv_path)

            if 0 < self.train_size <= 1:
                self.train_size = int(len(train_data) * self.train_size)

            train_data = train_data if self.train_size == 1 else train_data.shuffle(seed=42).select([i for i in list(range(int(self.train_size)))])
            val_data = val_data if self.val_size == 1 else val_data.shuffle(seed=42).select([i for i in list(range(int(self.val_size)))])
            test_data = test_data if self.test_size == 1 else test_data.shuffle(seed=42).select([i for i in list(range(int(self.test_size)))])

            print("Training class split: " + str(sum(train_data['label'])/len(train_data['label'])))
            print("Test class split: " + str(sum(test_data['label'])/len(test_data['label'])))
        return train_data, val_data, test_data

    def runner(self, output_path, train_bs, eval_bs, num_epochs, dataset_name, device_batch_size, lr=5e-05,
               logging_perc = -1, save_strategy = 'epoch', evaluation_strategy='epoch',
               load_best_model_at_end = False, no_cuda=False, eval_steps=-1, data_path=None, run=0, save_total_limit=1,
               metric_for_best_model='loss'):

        train_data, val_data, test_data = self.load_save_dataset(data_path=data_path,
                                                                 dataset_name=dataset_name,
                                                                 run=run,
                                                                 output_path=output_path)

        tokenized_train = train_data.map(self.tokenize, batched=True, batch_size=train_bs)
        # tokenized_val = val_data.map(self.tokenize, batched=True, batch_size=eval_bs)
        tokenized_test = test_data.map(self.tokenize, batched=True, batch_size=eval_bs)


        # Defaults if parameters shouldnt be interpreted as ranges
        if logging_perc == -1:
            logging_perc = 500
        if eval_steps == -1:
            eval_steps = None

        training_args = TrainingArguments(output_dir=output_path,
                                          learning_rate=lr,
                                          do_train=True,
                                          per_device_train_batch_size=device_batch_size,
                                          per_device_eval_batch_size=device_batch_size,
                                          num_train_epochs=num_epochs,
                                          evaluation_strategy=evaluation_strategy,
                                          save_strategy=save_strategy,
                                          load_best_model_at_end=load_best_model_at_end,
                                          save_total_limit=save_total_limit,
                                          metric_for_best_model=metric_for_best_model,
                                          weight_decay=0.01,
                                          eval_steps=eval_steps,
                                          logging_steps=logging_perc,
                                          no_cuda=no_cuda)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self._tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def prepare_laplace(self, output_path, train_bs, eval_bs, dataset_name, train_device_batch_size, eval_device_batch_size,
                        lr=5e-05, data_path = None, run=0):
        """

        :param output_path: Ouput path, for compatibility with other function calls, this function does not save
        :param train_bs: training batch size
        :param eval_bs: eval batch size
        :param dataset_name: (str) name of the dataset
        :param device_batch_size: per device batch size
        :return: None
        """

        train_data, val_data, test_data = self.load_save_dataset(data_path=data_path, dataset_name=dataset_name, run=run)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenized_train = train_data.map(self.tokenize, batched=True, batch_size=train_bs)
        tokenized_val = val_data.map(self.tokenize, batched=True, batch_size=eval_bs)
        # tokenized_test = test_data.map(self.tokenize, batched=True, batch_size=eval_bs)

        training_args = TrainingArguments(output_dir=output_path,
                                          learning_rate=lr,
                                          do_train=True,
                                          per_device_train_batch_size=train_device_batch_size,
                                          per_device_eval_batch_size=eval_device_batch_size,
                                          num_train_epochs=1,
                                          evaluation_strategy="epoch",
                                          save_strategy="epoch",
                                          weight_decay=0.01,
                                          laplace=True,
                                          no_cuda=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self._tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics
        )

        epoch_iterator, model, trainer = trainer.train()
        model = model.to(device)
        self.model = model
        return epoch_iterator, trainer, tokenized_val



def prepare_sentiment_classifier(args, model_name="distilbert-base-uncased"):
    if args.dataset_name == 'imdb' or args.dataset_name == 'sst2':
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        if hasattr(args, 'model_path'):
            model_path_or_name = args.model_path
        else:
            model_path_or_name = model_name

        sentiment_classifier = SentimentClassifier(model_path_or_name,
                                                   id2label=id2label,
                                                   label2id=label2id,
                                                   train_size=args.train_size,
                                                   val_size=args.val_size,
                                                   test_size=args.test_size,
                                                   dataset_name=args.dataset_name)

    else:
        sentiment_classifier = SentimentClassifier(model_name,
                                                   train_size=args.train_size,
                                                   val_size=args.val_size,
                                                   test_size=args.test_size,
                                                   dataset_name=args.dataset_name)

    return sentiment_classifier


def run_dataramping(args, sentiment_classifier=None, num_steps=10):
    train_sizes = np.linspace(0.1, 1.0, num_steps)
    num_epochs = [1 / x for x in train_sizes]

    for train_size, num_epoch in zip(train_sizes, num_epochs):
        args.train_size = train_size
        if sentiment_classifier is None:
            sentiment_classifier = prepare_sentiment_classifier(args)
        sentiment_classifier.runner(output_path=args.output_path,
                                    data_path=args.data_path,
                                    train_bs=args.train_batch_size,
                                    eval_bs=args.eval_batch_size,
                                    num_epochs=num_epoch,
                                    dataset_name=args.dataset_name,
                                    device_batch_size=args.device_batch_size,
                                    lr=args.learning_rate,
                                    logging_perc=args.logging_perc,
                                    save_strategy=args.save_strategy,
                                    evaluation_strategy=args.evaluation_strategy,
                                    load_best_model_at_end=args.load_best_model_at_end,
                                    save_total_limit=args.save_total_limit,
                                    no_cuda=args.no_cuda,
                                    eval_steps=args.eval_steps,
                                    run=args.run
                                    )


def prepare_and_run_sentiment_classifier(args, sentiment_classifier=None):
    if sentiment_classifier is None:
        sentiment_classifier = prepare_sentiment_classifier(args)

    sentiment_classifier.runner(output_path=args.output_path,
                                data_path=args.data_path,
                                train_bs=args.train_batch_size,
                                eval_bs=args.eval_batch_size,
                                num_epochs=args.num_epochs,
                                dataset_name=args.dataset_name,
                                device_batch_size=args.device_batch_size,
                                lr=args.learning_rate,
                                logging_perc = args.logging_perc,
                                save_strategy = args.save_strategy,
                                evaluation_strategy=args.evaluation_strategy,
                                metric_for_best_model=args.metric_for_best_model,
                                load_best_model_at_end=True,
                                no_cuda=args.no_cuda,
                                eval_steps=args.eval_steps,
                                run=args.run)

    return None


def construct_laplace(sent_class, laplace_cfg, args):
    train_loader, trainer = sent_class.prepare_laplace(output_path=args.output_path,
                                                       data_path=args.data_path,
                                                       train_bs=args.train_batch_size,
                                                       eval_bs=args.eval_batch_size,
                                                       dataset_name=args.dataset_name,
                                                       device_batch_size=args.device_batch_size,
                                                       lr=args.learning_rate,
                                                       run=args.run)

    if not isinstance(sent_class.model, Extension):
        model = Extension(sent_class.model)
    else:
        model = sent_class.model

    for idx, module_name in enumerate(laplace_cfg.module_names):
        if module_name.split(".")[0] != 'model':
            laplace_cfg.module_names[idx] = ".".join(('model', module_name))

    partial_constructor = PartialConstructor(model, module_names='model.classifier')
    partial_constructor.select()

    la = lp.Laplace(model, laplace_cfg.ml_task,
                    subset_of_weights=laplace_cfg.deprecated_subset_of_weights,
                    hessian_structure=laplace_cfg.hessian_structure,
                    prior_precision=laplace_cfg.prior_precision)

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
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_epochs", type=float, default=1)
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--train_size", type=float, default=1) # 1 gives full dataset
    parser.add_argument("--val_size", type=float, default=1) # 1 gives full dataset
    parser.add_argument("--test_size", type=float, default=1) # 1 gives full dataset
    parser.add_argument("--device_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-05)
    parser.add_argument('--swag_cfg', default=None)
    parser.add_argument('--la_cfg', default=None)
    parser.add_argument('--logging_perc',type = float, default = -1) # -1 for default from transformers
    parser.add_argument('--eval_steps', type = float, default = -1) # -1 for default from transformers
    parser.add_argument('--save_strategy', default = 'epoch') # 'epoch' for default from transformers
    parser.add_argument('--evaluation_strategy', default = 'epoch')
    parser.add_argument('--no_cuda', type=ast.literal_eval, default=False)
    parser.add_argument('--dataramping', type=ast.literal_eval, default=False)
    parser.add_argument('--load_best_model_at_end', type=ast.literal_eval, default=False)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--metric_for_best_model', type=str, default='loss')
    parser.add_argument('--run', type=int, default=0)

    args = parser.parse_args()

    if int(args.train_size)-args.train_size == 0:
        args.train_size = int(args.train_size)

    if args.dataramping:
        run_dataramping(args, num_steps=10)
    else:
        prepare_and_run_sentiment_classifier(args)

    # elif args.swag:
    #     cfg = Namespace(**yaml.safe_load(open(
    #         'swag_cfg.yaml' if args.swag_cfg is None else args.swag_cfg, 'r')
    #     ))
    #     if not args.train:
    #         raise NotImplementedError("Swag model not yet implemented with evaluation protocol")
    #
    #     sentiment_classifier = prepare_sentiment_classifier(args)
    #     sentiment_classifier = construct_swag(sentiment_classifier, cfg)
    #     prepare_and_run_sentiment_classifier(args, sentiment_classifier)
    #     save_object_(sentiment_classifier.model, args.output_path, 'swag', args.swag_cfg)
    #
    # elif args.laplace:
    #     cfg = Namespace(**yaml.safe_load(open(
    #         'laplace_cfg.yaml' if args.la_cfg is None else args.la_cfg, 'r')
    #     ))
    #     sentiment_classifier = prepare_sentiment_classifier(args)
    #     la = construct_laplace(sentiment_classifier, cfg, args)
    #
    #     save_object_(la, args.output_path, 'laplace', args.la_cfg)
