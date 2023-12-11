import ast
import copy
import os
import pickle

import torch
from datasets import load_dataset, Dataset
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
from SentimentAnalysis.PartialConstructor import PartialConstructor, PartialConstructorSwag, Truncater, Extension
import torch.nn as nn
from Laplace.laplace import Laplace
import uncertainty_toolbox as uct
from torch.nn import BCELoss, CrossEntropyLoss
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

font = {'family': 'serif',
            'size': 15,
            'serif': 'cmr10'
            }
mpl.rc('font', **font)
mpl.rc('legend', fontsize=15)
mpl.rc('axes', labelsize=19)
map_color = 'tab:red'
stochastic_color = 'tab:green'
point_err_color = 'tab:blue'
plt.rcParams['axes.unicode_minus'] = False


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


def read_file(path):
    pcl = pickle.load(open(path, 'rb'))
    return pcl
def run_evaluator_again(predictions, labels):

    m = nn.Softmax(dim = 1)
    predictions = m(predictions)
    evaluator = Evaluator(predictions, labels)
    evaluator.get_all_metrics()
    breakpoint()



class RampingExperiments:

    def __init__(self, ramping_exp_path, metric = 'nll'):
        self.ramping_exp_path = ramping_exp_path
        self.metric = metric
        self.color = point_err_color

    def find_files(self, path = None):
        path = self.ramping_exp_path if path is None else path
        files = [os.path.join(path, p) for p in os.listdir(path)]
        run_numbers_and_paths = []
        for file in files:
            if 'run_' in os.path.basename(file) and os.path.isdir(file):
                run_number = int(os.path.basename(file).split("_")[-1])
                pp = os.path.join(file, f'run_number_{run_number}.pkl')
                if not os.path.exists(pp):
                    pp = os.path.join(file, f'run_number_{0}.pkl')
                    if not os.path.exists(pp):
                        raise ValueError("Could not find path")
                run_numbers_and_paths.append((run_number,pp))

        return sorted(run_numbers_and_paths)
    def get_metrics_from_file(self, file, has_seen_softmax = True):

        evaluation = read_file(file)
        if 'results' in evaluation:
            results = evaluation['results']
        else:
            results = evaluation
        if not isinstance(results, dict):
            if has_seen_softmax:
                eval_ = Evaluator(results.predictions, results.labels,
                                             has_seen_softmax=has_seen_softmax)

                results_recalculated = eval_.get_all_metrics()
            else:
                results_recalculated = results.results
            results_recalculated['modules'] = 1
            res = {k: [v] for k, v in results_recalculated.items() if not isinstance(v, (list, tuple))}
            return res
        modules = list(results.keys())
        res = {k: [] for k in results[modules[0]].results.keys()}
        print(file, modules)
        for module in modules:
            if has_seen_softmax:
                eval_ = Evaluator(results[module].predictions, results[module].labels,
                                             has_seen_softmax=has_seen_softmax)
                results_recalculated = eval_.get_all_metrics()
            else:
                results_recalculated = results[module].results

            for k, v in results_recalculated.items():
                res[k].append(v)
        res['modules'] = modules
        return res

    def get_metrics_from_all_files(self, path = None, has_seen_softmax = True):

        run_number_and_paths = self.find_files(path)
        results = {}
        for run_number, path in run_number_and_paths:
            results[run_number] = self.get_metrics_from_file(path, has_seen_softmax=has_seen_softmax)

        return results

    def get_specific_results(self, results, key, map_path = None):

        df = pd.DataFrame()
        module_holder, results_holder = [], []

        for run, v in results.items():
            modules = v['modules']
            module_holder+=modules
            results_holder+= v[key]

        if map_path is not None:
            map_results = self.include_map(map_path)
            results_holder += map_results[key]
            module_holder += [0]*len(map_results[key])

        df[key] = results_holder
        df['modules'] = module_holder
        return df


    def plot_result(self, df,key, ax = None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
            show_ = True
        else:
            show_ = False

        def errorbar_normal(x):
            mean = np.mean(x)
            sd = np.std(x)
            return mean - 1.96 * sd, mean + 1.96 * sd
        # errorbar_func = lambda x: np.mean(
        sns.pointplot(errorbar=errorbar_normal,
                      data=df, x="modules", y=key,
                      join=False,
                      capsize=.30,
                      markers="d",
                      scale=1.0,
                      err_kws={'linewidth': 0.7}, estimator=np.mean,
                      color=self.color,
                      label=key.split("_")[0] + " " + " ".join(self.experiment_name.split("_")),
                      ax=ax)

        if show_ :
            plt.show()


    def get_and_plot(self, path = None, map_path = None, has_seen_softmax = True, ax = None):
        if path is None:
            path = self.ramping_exp_path

        self.experiment_name = os.path.basename(path)
        results = self.get_metrics_from_all_files(path, has_seen_softmax = has_seen_softmax)
        key = self.metric

        df = self.get_specific_results(results, key, map_path)
        # df = df[df['modules'] <= 11]
        self.plot_result(df, key, ax = ax)

    def include_map(self, path):

        map_paths = [os.path.join(path, p) for p in os.listdir(path)]
        res_ = {}
        for pa in map_paths:
            results = read_file(pa)
            results = results['results']
            for k, v in results.results.items():
                if k not in res_:
                    res_[k] = []
                res_[k].append(v)

        return res_



def evaluate_loss_with_only_wrong(predictions, labels):

    where = np.argmax(predictions, -1) == labels
    return predictions[~where], labels[~where]

class SmallerDataloaderForOptimizing:
    def __init__(self, other_dataloader, indices):
        self.other_dataloader = other_dataloader
        self.full_dataloader = iter(other_dataloader)
        self.indices = sorted(indices)
        self.current_index = -1
        self.dataset = Dataset.from_dict(self.other_dataloader.dataset[indices])

    def __next__(self):
        if len(self.indices) > self.current_index+1:
            current_idx = self.current_index
            while self.indices[current_idx] != self.indices[self.current_index + 1]:
                _ = next(self.full_dataloader)
                current_idx += 1
            batch = next(self.full_dataloader)
            self.current_index += 1
            return batch
        else:
            self.full_dataloader = iter(self.other_dataloader)
            raise StopIteration

    def __iter__(self):
        self.current_index = -1
        return self

def get_smaller_dataloader(dataloader, indices):
    dataloader = SmallerDataloaderForOptimizing(dataloader, indices)
    return dataloader

if __name__ == '__main__':



    path = r'C:\Users\45292\Documents\Master\SentimentClassification\Laplace\operator_norm_ramping'
    path = r'C:\Users\45292\Documents\Master\SentimentClassification\Laplace\random_ramping'
    # path = r'C:\Users\45292\Documents\Master\SentimentClassification\SWAG\random_ramping'
    map_path = r'C:\Users\45292\Documents\Master\SentimentClassification\Laplace\map'
    path = r'C:\Users\45292\Documents\Master\SentimentClassification\Laplace\last_layer_prior'
    fig, ax = plt.subplots(1, 1)
    plotter = RampingExperiments(path, 'ECE')
    plotter.get_and_plot(path = path, has_seen_softmax = True, ax = ax, map_path=map_path)
    plotter.color = 'tab:orange'
    # plotter.get_and_plot(path = path_, has_seen_softmax=True, ax = ax, map_path=map_path)
    plt.gcf().subplots_adjust(left=0.16)
    ylims = ax.get_ylim()
    diff = (ylims[1] - ylims[0]) * 0.3 + ylims[1]

    ax.set_ylim((ylims[0], diff))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
              ncol=1, fancybox=True, shadow=True)

    plt.show()



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
