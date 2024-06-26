import ast
import copy
import os
import pickle

import matplotlib.text
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
from sklearn.calibration import CalibrationDisplay, calibration_curve
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



class Point:

    def __init__(self, x, y, color, marker = 'o', label = None, name = None):
        self.x = x
        self.y = y
        self.color = color
        self.marker = marker
        self.label = label
        self.name = name
    def draw(self, ax):
        ax.plot(self.x, self.y, marker = self.marker, color = self.color, label = self.label,
                markersize = 11)
        return ax

class Line:

    def __init__(self, point_one, point_two):
        self.point_one = point_one
        self.point_two = point_two

    def draw(self, ax):

        if self.point_one.color != self.point_two.color:
            raise ValueError("Points have different color, likely a mistake somewhere else")

        ax.plot([self.point_one.x, self.point_two.x], [self.point_one.y, self.point_two.y],
                color = self.point_one.color, linewidth = 2)

        return ax

class LayerCombiner:

    def __init__(self,  layer_one, layer_two):
        self.layer_one = layer_one
        self.layer_two = layer_two

        self.combining_lines = []
        for point_one in self.layer_one:
            for point_two in self.layer_two:
                if point_one.name == point_two.name:
                    self.combining_lines.append(Line(point_one, point_two))

    def draw(self, ax):
        for line in self.combining_lines:
            ax = line.draw(ax)
        return ax

class RankingPlotter:

    def __init__(self, rankings, names, modules):

        self.rankings = rankings
        self.names = names
        self.modules = modules
        self.point_type = 'o'
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']

        self.layers = []
        for idx, mod in enumerate(self.modules):
            self.layers.append(self.make_layer(idx))

        self.layer_combiners = []
        for layer_one, layer_two in zip(self.layers[:-1], self.layers[1:]):
            self.layer_combiners.append(LayerCombiner(layer_one, layer_two))


    def draw(self,  ax):

        for layer in self.layers:
            for point in layer:
                ax = point.draw(ax)
        for combiner in self.layer_combiners:
            ax = combiner.draw(ax)

        return ax
    def make_layer(self, idx):
        counter_to_pos_two = {i: {0: -0.2, 1: 0.2} for i in range(5)}
        counter_to_pos_three = {i : {0: -0.2, 1: 0, 2: 0.2} for i in range(5)}
        counter_two = {i: 0 for i in range(5)}
        counter_three = {i: 0 for i in range(5)}
        layer = []
        scores = self.rankings[idx]

        for i, name in enumerate(self.names):
            score = scores[i]
            if np.sum(score == scores) == 1:
                x_pos = self.modules[idx]
            elif np.sum(score == scores) == 2:
                x_pos = self.modules[idx] + counter_to_pos_two[int(score)][counter_two[int(score)]]
                counter_two[int(score)] += 1
            elif np.sum(score == scores) == 3:
                x_pos = self.modules[idx] + counter_to_pos_three[int(score)][counter_three[int(score)]]
                counter_three[int(score)] += 1
            else:
                raise ValueError("Something is terribly wrong")
            label = name if idx == 0 else None
            layer.append(Point(x_pos, score, self.colors[i], self.point_type, label, name))
        return layer



def save_laplace(save_path, laplace_cfg, **kwargs):
    pass


class MemoryCalculator:
    def __init__(self, num_modules = None, percentage = None, module_type = None, laplace = False, SWAG=False):

        self.num_modules = num_modules
        self.percentage = percentage
        if self.percentage is not None:
            self.percentage = self.percentage / 100
        self.module_type = module_type
        self.laplace = laplace
        self.SWAG = SWAG

        self.num_linear_weights = 43100930
        self.num_small_linears = 25
        self.num_mlp_linears = 12
        self.num_classifiers = 1
        self.small_in = 768
        self.small_out = 768
        self.big_in = 768
        self.big_out = 3072
        self.memory_calculator = self.construct_memory_calculator()
        self.true_gb_number = 1073741824
    def memory_calculator_skfac(self):

        small_a_and_b = 2 * (self.small_in*self.percentage * self.small_out*self.percentage) * self.num_small_linears
        big_a_and_b = ((self.big_in * self.percentage * self.big_in * self.percentage) +
                       self.big_out * self.percentage * self.big_out * self.percentage) * self.num_mlp_linears

        num_params = ((self.small_in*self.percentage * self.small_out*self.percentage) * self.num_small_linears +
                      self.big_in * self.percentage * self.big_out * self.percentage)

        number_of_floats_kfac = small_a_and_b + big_a_and_b
        number_of_floats_jac = num_params * 2
        return self.floats_to_gb(number_of_floats_kfac + number_of_floats_jac)

    def floats_to_gb(self, num_floats):
        return num_floats


    def memory_calculator_modules_la(self, num_modules = None, module_type = None):

        num_mod = self.num_modules if num_modules is None else num_modules
        module_type = self.module_type if module_type is None else module_type
        if module_type == 'mlp':
            number_of_floats_kfac = num_mod * (self.big_out**2 + self.big_in**2)
            number_of_floats_jac = num_mod * self.big_out * self.big_in + num_mod/2 * self.big_in**2 + num_mod/2 * self.big_out**2

        elif module_type == 'attn':
            number_of_floats_kfac = num_mod * (self.small_in**2*2 + self.small_out ** 2)
            number_of_floats_jac = num_mod * self.small_in * self.small_out + num_mod * self.small_in**2

        elif module_type == 'random':
            num_mod_attn = self.num_small_linears / (self.num_small_linears + self.num_mlp_linears) * self.num_modules
            num_mod_mlp = self.num_mlp_linears / (self.num_small_linears + self.num_mlp_linears) * self.num_modules
            return self.memory_calculator_modules_la(num_mod_attn, 'attn') +\
                self.memory_calculator_modules_la(num_mod_mlp, 'mlp')
        else:
            raise ValueError("")

        return self.floats_to_gb(number_of_floats_kfac + number_of_floats_jac)


    def memory_calculator_modules_swag(self):

        num_params = self.num_modules / (self.num_mlp_linears + self.num_small_linears) * self.num_linear_weights
        num_floats = 21 * num_params
        return self.floats_to_gb(num_floats)

    def memory_calculator_percentage_swag(self):
        pass

    def construct_memory_calculator(self):

        if self.laplace:
            if self.module_type == 'last_layer':
                return lambda: self.floats_to_gb((768*2)**2 + 768*2)

            elif self.percentage is not None:
                return self.memory_calculator_skfac
            else:
                return self.memory_calculator_modules_la
        elif self.SWAG:
            if self.percentage is not None:
                return self.memory_calculator_percentage_swag
            else:
                return self.memory_calculator_modules_swag



class Evaluator:
    def __init__(self, predictions, labels, has_seen_softmax=False):

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

    def compute_torch_metrics(self, results=None):

        if results is None:
            results = {}
        results['nll'] = self.nll_metric(self.predictions, self.labels).item()

        for key, val in self.torchmetrics_.items():
            results[key] = val(self.predictions, self.labels).item()

        return results

    def compute_standard_metrics(self, results=None, copy=True):

        if results is None:
            results = {}

        if copy:
            predictions, labels = self.predictions.clone().numpy(), self.labels.clone().numpy()
        else:
            predictions, labels = self.predictions.numpy(), self.labels.numpy()

        for key, val in self.all_standard_metrics.items():
            results[key] = val(labels, predictions.argmax(-1))

        return results

    def get_all_metrics(self, copy=False, override=True):

        results = self.compute_torch_metrics()
        results = self.compute_standard_metrics(results=results, copy=copy)
        if len(self.results) == 0 or override:
            self.results = results
        return results


def evaluate_laplace(la, trainer: Trainer, eval_dataset=None):
    eval_dataset = trainer.get_eval_dataloader(eval_dataset=eval_dataset)

    la.model.eval()
    predictions, labels = [], []
    for step, x in enumerate(eval_dataset):
        output = la(x)
        predictions.append(output)
        labels.append(x['labels'])

    predictions, labels = (torch.cat(predictions, dim=0).detach().cpu(),
                           torch.cat(labels, dim=0).detach().cpu())

    evaluator = Evaluator(predictions, labels, has_seen_softmax=True)
    evaluator.get_all_metrics()
    return evaluator


def evaluate_swag(swag: PartialConstructorSwag, trainer: Trainer, eval_dataset=None):
    eval_dataset = trainer.get_eval_dataloader(eval_dataset=eval_dataset)
    swag.eval()

    predictions, labels = [], []
    for step, x in enumerate(eval_dataset):
        output = swag.predict_mc(**x)
        predictions.append(output)
        labels.append(x['labels'])

    predictions, labels = (torch.cat(predictions, dim=0).detach().cpu(),
                           torch.cat(labels, dim=0).detach().cpu())

    evaluator = Evaluator(predictions, labels, has_seen_softmax=True)
    evaluator.get_all_metrics()
    return evaluator


def evaluate_map(model, trainer: Trainer, eval_dataset=None):
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


def evaluate_loss(predictions, labels, use_softmax=False):
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
    m = nn.Softmax(dim=1)
    predictions = m(predictions)
    evaluator = Evaluator(predictions, labels)
    evaluator.get_all_metrics()
    breakpoint()


class MultipleRampingExperiments:

    def __init__(self, ramping_exp_paths,
                 ramping_exp_names=None,
                 map_path=None,
                 metric='nll',
                 sublayer_ramping=False,
                 method='la'):
        self.ramping_exp_paths = ramping_exp_paths
        self.ramping_exp_names = ramping_exp_names
        self.ramping_exp_names = self.get_names_from_paths()

        if isinstance(map_path, list):
            self.map_path = map_path
        else:
            self.map_path = [map_path]*len(self.ramping_exp_paths)


        self.metric = metric
        self.sublayer_ramping = sublayer_ramping
        self.dataset_name = self.get_dataset_name(self.ramping_exp_paths[0])

        if method in ['la', 'laplace', 'Laplace']:
            self.method_name = 'Laplace'
        elif method in ['sw', 'swa', 'swag', 'SWAG']:
            self.method_name = 'SWAG'
        else:
            self.method_name = method

        self.ramping_experiments = {
            path: RampingExperiments(path, metric=metric) for path in self.ramping_exp_paths
        }
        self.path_to_map_path = {path: map_p for path, map_p in zip(self.ramping_exp_paths, self.map_path)}
        self.path_to_names = {path: name for path, name in zip(self.ramping_exp_paths, self.ramping_exp_names)}

        self.exp_number_to_colors = ['tab:blue', 'tab:orange', 'tab:red',
                                     'tab:green', 'tab:purple', 'tab:brown',
                                     'tab:pink', 'tab:gray', 'tab:olive',
                                     'tab:cyan']

        self.metric_to_label_metric = {
            'accuracy_score': 'Accuracy',
            'nll': 'NLL',
            'balanced_accuracy_score': 'Balanced Accuracy',
            'f1': 'F1 score',
            'ECE': 'ECE', 'MCE': 'MCE', 'RMSCE': 'RMSCE'
        }


    def calc_temperature_scaling(self, map_path, val_path):
        ramping_exp = self.ramping_experiments[self.ramping_exp_paths[0]]
        preds, labels, nll, ece = ramping_exp.apply_temp_scaling(map_path, val_path)
        return nll.item(), ece.item()


    def find_best_and_make_bold(self, df, column, num_indices=3):

        values = list(df[column])

        if self.metric in ['nll', 'ECE', 'MCE', 'ece', 'NLL', 'RMSCE']:
            use_min = True
        else:
            use_min = False

        for idx, val in enumerate(values):
            if np.isnan(val):
                if use_min:
                    values[idx] = np.inf
                else:
                    values[idx] = -np.inf

        if use_min:
            best = np.argmin(values)
        else:
            best = np.argmax(values)

        new_values = []
        for idx, val in enumerate(values):
            if idx == best:
                format_ = f"\\textbf{'{' + '{:.2f}'.format(val) + '}'}"
            else:
                if np.abs(val) == np.inf:
                    format_ = "-"
                else:
                    format_ = '{:.2f}'.format(val)
            new_values.append(format_)
        df[column] = new_values
        return df

    def write_latex_table(self, bold_direction='modules'):
        dataframes = self.get_dfs()
        dfs = []
        df_combined = pd.DataFrame()
        max_num_modules = 0
        all_modules = set()
        for key, val in dataframes.items():
            all_modules = all_modules.union(set(val['modules'].unique()))

        all_modules = sorted(list(all_modules))
        for idx, key in enumerate(self.ramping_experiments.keys()):
            df = dataframes[key]
            name = self.path_to_names[key]
            medians = []
            for module in all_modules:
                if any(df['modules'] == module):
                    medians.append(df[df['modules'] == module][self.metric].median())
                else:
                    medians.append(np.nan)
            df_combined[name] = medians

        if bold_direction == 'modules':
            for column in df_combined.columns:
                df_combined = self.find_best_and_make_bold(df_combined, column)

        df_combined = df_combined.transpose()
        df_combined = df_combined.rename(
            columns={i: f"{int(perc)}" for i, perc in zip(df_combined.columns, all_modules)})

        if bold_direction == 'method':
            for column in df_combined.columns:
                df_combined = self.find_best_and_make_bold(df_combined, column)

        print(r"\begin{table}", "\n", '\centering \n', '\caption{} \n',
              df_combined.to_latex(index=True, float_format="{:.2f}".format, escape=False),
              '\label{} \n', r"\end{table}", "\n")


    def find_best_measure(self, other_path, num_modules = None, max_mod = None, return_uncertainty = False):
        ramping_experiment = RampingExperiments(other_path, metric=self.metric)
        results = ramping_experiment.get_metrics_from_all_files(has_seen_softmax=True)

        df = ramping_experiment.get_specific_results(results, self.metric)
        best_score, best_mod = None, None
        medians, modules, uncertainties = [], [], []
        if num_modules is None:
            all_mods = df['modules'].unique()
            if max_mod:
                all_mods = [mod for mod in df['modules'].unique() if mod < max_mod]
            for mod in all_mods:
                scores = df[df['modules'] == mod][self.metric]
                medians.append(scores.median())
                uncertainties.append(list(scores))
                modules.append(mod)
                if self.metric not in ['nll', 'ECE', 'RMSCE', 'MCE']:
                    best_score = np.max(medians)
                    best_mod = modules[np.argmax(medians)]
                    uncertainty = uncertainties[np.argmax(medians)]
                else:
                    best_score = np.min(medians)
                    best_mod = modules[np.argmin(medians)]
                    uncertainty = uncertainties[np.argmin(medians)]
        else:
            scores = df[df['modules'] == num_modules][self.metric]
            best_score = scores.median()
            best_mod = num_modules
            uncertainty = list(scores)
        if return_uncertainty:
            return best_score, best_mod, uncertainty
        else:
            return best_score, best_mod   # Not really sure I need to return best_mod here, but it never hurts
    def draw_line_at_best(self, other_path, ax, name=None, color=None, num_modules=None, best_mod=True,
                          set_point_instead=False, sublayer_ramping=False, return_stuff = False):

        ramping_experiment = RampingExperiments(other_path, metric=self.metric)
        results = ramping_experiment.get_metrics_from_all_files(has_seen_softmax=True)
        df = ramping_experiment.get_specific_results(results, self.metric)
        medians, modules = [], []
        if num_modules is None:
            for mod in df['modules'].unique():
                scores = df[df['modules'] == mod][self.metric]
                medians.append(scores.median())
                modules.append(mod)
                if self.metric not in ['nll', 'ECE', 'RMSCE', 'MCE']:
                    best_score = np.max(medians)
                    best_mod = int(modules[np.argmax(medians)])
                else:
                    best_score = np.min(medians)
                    best_mod = int(modules[np.argmin(medians)])

        else:
            scores = df[df['modules'] == num_modules][self.metric]
            best_score = scores.median().item()

            best_mod = int(num_modules)

        if return_stuff:
            return best_score, best_mod, scores

        if  'Last layer' not in name:
            if sublayer_ramping:
                best_mod = np.round(self.percentile_to_actual_percentile(best_mod), 3)

            label_name = name + " " + str(best_mod)
        else:
            label_name = name

        ylims = ax.get_ybound()
        difference = ylims[1] - ylims[0]
        if best_score >= ylims[1] - difference / 10:
            ax.set_ybound(ylims[0], best_score + difference / 5)

        for line in ax.lines:
            x,y = line.get_data()
            ylims = ax.get_ybound()
            relation = (ylims[1] - ylims[0]) / 4
            if np.any(y > (ylims[1] - relation)):
                new_bound = ylims[1] + relation * 4/7
                ax.set_ybound(ylims[0], new_bound)
        if best_score <= ylims[0] + difference / 10:
            ax.set_ybound(best_score - difference / 5, ylims[1])
        if set_point_instead:
            scores = df[df['modules'] == num_modules]
            num_floats = (768*2)**2 + (762*4)
            scores['modules'] = [num_floats for _ in range(len(scores))]
            sns.pointplot(errorbar=lambda x: np.percentile(x, [25, 75]),
                          data=scores, x="modules", y=self.metric,
                          join=False,
                          capsize=.30,
                          markers="d",
                          scale=1.0,
                          err_kws={'linewidth': 0.7}, estimator=np.median,
                          color=color,
                          label=None,
                          ax=ax)

            # ticks = ax.get_xmajorticklabels()
            # ticks[-1].set_text('100')
            # ax.set_xticklabels(ticks)
        else:

            ax.axhline(y=best_score, linestyle='--', linewidth=2.5, alpha=0.7,
                       color='tab:red' if color is None else color, label=label_name)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                      ncol=1, fancybox=True, shadow=True)
        return ax

    def get_dataset_name(self, path):
        if 'imdb' in path.lower():
            return 'IMDb'
        elif 'sst' in path.lower():
            return 'SST2'

        elif 'rte' in path.lower():
            return 'RTE'
        elif 'mrpc' in path.lower():
            return 'MRPC'
        else:
            return ""

    def get_dfs(self):

        dfs = {}
        for key, val in self.ramping_experiments.items():
            dfs[key] = val.get_df(map_path=self.path_to_map_path[key], has_seen_softmax=True)

        return dfs

    def plot_all(self, fig=None, ax=None, subset_ = None, spec_subset = None,
                 use_num_floats = False, return_the_effing_dataframe = False):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        dataframes = self.get_dfs()

        def errorbar_normal(x):
            mean = np.mean(x)
            sd = np.std(x)
            return mean - 1.96 * sd, mean + 1.96 * sd

        # errorbar_func = lambda x: np.mean(
        error_bar_percentile = lambda x: np.percentile(x, [25, 75])
        for idx, key in enumerate(self.ramping_experiments.keys()):
            df = dataframes[key]
            if subset_ is not None:

                # df = df[df['modules'] != sorted(df['modules'])[7]]
                for mod in df['modules'].unique():
                    if subset_ <= mod:
                        df = df[df['modules'] != mod]

            if use_num_floats:
                new_df = df.copy(deep=True)
                for mod in df['modules']:
                    memory_calculator = MemoryCalculator(percentage=mod, laplace = True)
                    num_floats = memory_calculator.memory_calculator()
                    new_df.loc[df['modules'] == mod, 'modules'] = num_floats
                df = new_df

            if return_the_effing_dataframe:
                return df
            name = self.path_to_names[key]
            color = self.exp_number_to_colors[idx]

            sns.pointplot(errorbar=error_bar_percentile,
                          data=df, x="modules", y=self.metric,
                          join=False,
                          capsize=.30,
                          markers="d",
                          markersize = 50,
                          scale=1.5,
                          err_kws={'linewidth': 1.5}, estimator=np.median,
                          color=color,
                          label=name,
                          ax=ax)

        self.set_proper_axis_labels(ax)
        ax.set_title(self.method_name + " " + self.dataset_name + " " + self.metric_to_label_metric[self.metric])
        self.set_bounds(ax)
        return fig, ax

    def set_bounds(self, ax):
        plt.gcf().subplots_adjust(left=0.16)
        ylims = ax.get_ylim()
        additional_top = (ylims[1] - ylims[0]) * 0.35 + ylims[1]
        ax.set_ylim((ylims[0], additional_top))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                  ncol=1, fancybox=True, shadow=True)
        return ax

    def percentile_to_actual_percentile(self, percentile):
        num_params_mlp = lambda x: 11 * 768 / 100 * x * 3072 / 100 * x
        num_params_attn = lambda x: 27 * 768 / 100 * x * 768 / 100 * x

        percentage_of_params = ((num_params_attn(percentile) + num_params_mlp(percentile))
                                / (num_params_mlp(100) + num_params_attn(100))) * 100
        return percentage_of_params

    def set_proper_axis_labels(self, ax):
        ticks = ax.get_xmajorticklabels()
        numbers = [float(t._text) for t in ticks]

        if self.sublayer_ramping:
            def percentile_to_actual_percentile(percentile):
                num_params_mlp = lambda x: 11 * 768 / 100 * x * 3072 / 100 * x
                num_params_attn = lambda x: 27 * 768 / 100 * x * 768 / 100 * x

                percentage_of_params = ((num_params_attn(percentile) + num_params_mlp(percentile))
                                        / (num_params_mlp(100) + num_params_attn(100))) * 100
                return percentage_of_params

            numbers = [str(np.round(percentile_to_actual_percentile(number), 3)) for number in numbers]
            x_label = 'Percentages'

        else:
            numbers = [str(int(number)) for number in numbers]
            x_label = "Num. stoch. modules "

        def handle_str_number(number):
            numb = float(number)
            numb = np.round(numb, 2)
            return str(numb)
        y_label = self.metric_to_label_metric[self.metric]
        for number, tick in zip(numbers, ticks):
            tick.set_text(handle_str_number(number))

        ax.set_xticklabels(ticks)
        # ax.xaxis.set_major_formatter('{x:1.2f}')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax

    def get_names_from_paths(self):

        if self.ramping_exp_names is not None:
            return self.ramping_exp_names

        names = []
        for path in self.ramping_exp_paths:
            if 'run' not in path:
                name = os.path.basename(path).split("_")
                if 'prior' in name[-1].lower():
                    name = name[:-1]
                names.append(" ".join(name))

        return names


class RampingExperiments:

    def __init__(self, ramping_exp_path, metric='nll'):
        self.ramping_exp_path = ramping_exp_path
        self.metric = metric
        self.color = point_err_color

    def find_files(self, path=None):
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
                run_numbers_and_paths.append((run_number, pp))

        return sorted(run_numbers_and_paths)


    def get_predictions_from_file(self, file, key = "", softmax = True):

        evaluation = read_file(file)
        if 'results' in evaluation:
            results = evaluation['results']
        else:
            results = evaluation

        if key:
            results = results[key]
        softmax_ = nn.Softmax(dim = -1)
        predictions = results.predictions
        if predictions.sum(-1).mean() != 1 and softmax:
            predictions = softmax_(predictions)

        labels = results.labels
        return predictions, labels

    def apply_temp_scaling(self, test_path, val_path):

        test_predictions, test_labels = self.get_predictions_from_file(test_path)
        val_predictions, val_labels = self.get_predictions_from_file(val_path)

        self.train_temperature_scaling(val_predictions, val_labels)
        softmax_ = nn.Softmax(dim = -1)
        scaled_test = self.temperature_scaler(test_predictions)
        nll = self.temperature_scaler.loss_function(scaled_test, test_labels)
        ece = MulticlassCalibrationError(num_classes=2, n_bins=20)(scaled_test, test_labels)
        test_predictions = softmax_(self.temperature_scaler(test_predictions)).detach()
        return test_predictions, test_labels, nll, ece


    def train_temperature_scaling(self, predictions, labels):
        from TemperatureScaler import TemperatureScaler
        self.temperature_scaler = TemperatureScaler(predictions, labels)
        self.temperature_scaler.optimize(int(1e3))

    def collect_predictions_and_calculate_calib(self, key, map_path = "", val_path = ""):
        path = self.ramping_exp_path
        self.experiment_name = os.path.basename(path)

        run_number_and_paths = self.find_files(path)
        df = pd.DataFrame()
        prop_trues, prob_preds = [], []
        for run_number, path in run_number_and_paths:
            predictions, labels = self.get_predictions_from_file(path, key = key)
            prob_true, prob_pred = self.get_calib_(predictions, labels)
            prop_trues += list(prob_true)
            prob_preds += list(prob_pred)
            break

        if map_path:
            map_path = os.path.join(map_path, f"run_number_{run_number}_map.pkl")
            if not val_path:
                predictions_map, labels_map = self.get_predictions_from_file(map_path)
            else:
                val_path = os.path.join(val_path, f"run_number_{run_number}_map.pkl")
                predictions_map, labels_map, nll, ece = self.apply_temp_scaling(map_path, val_path)

            prob_true_map, prob_pred_map = self.get_calib_(predictions_map, labels_map)
            prob_true_map = list(prob_true_map)
            prob_pred_map = list(prob_pred_map)
            output_map = (prob_true_map, prob_pred_map, predictions_map)

        else:
            output_map = ()

        return (prob_true, prob_pred, predictions[:, 1]), output_map

    def get_calib_(self, predictions, labels):
        prob_true, prob_pred = calibration_curve(labels, predictions[:, 1], n_bins=20,strategy='quantile')
        return prob_true, prob_pred

    def plot_calibration(self, prob_true, prob_pred, predictions, ax = None, color = 'tab:orange', label = ""):
        calib_disp = CalibrationDisplay(prob_true, prob_pred,predictions)
        calib_disp.plot(ax = ax, color = color, label = label)


    def get_metrics_from_file(self, file, has_seen_softmax=True):

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
        modules = [mod for mod in modules if 'module_sel' not in str(mod)]
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

    def get_metrics_from_all_files(self, path=None, has_seen_softmax=True):

        run_number_and_paths = self.find_files(path)
        results = {}
        for run_number, path in run_number_and_paths:
            results[run_number] = self.get_metrics_from_file(path, has_seen_softmax=has_seen_softmax)

        return results

    def get_specific_results(self, results, key, map_path=None):

        df = pd.DataFrame()
        module_holder, results_holder = [], []

        for run, v in results.items():
            modules = [float(mod) for mod in v['modules']]
            module_holder += modules
            try:
                results_holder += v[key]
            except:
                breakpoint()
        if map_path is not None:
            map_results = self.include_map(map_path)
            results_holder += map_results[key]
            module_holder += [0] * len(map_results[key])

        df[key] = results_holder
        df['modules'] = module_holder
        return df

    def plot_result(self, df, key, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            show_ = True
        else:
            show_ = False

        def errorbar_normal(x):
            mean = np.mean(x)
            sd = np.std(x)
            return mean - 1.96 * sd, mean + 1.96 * sd

        # errorbar_func = lambda x: np.mean(
        error_bar_percentile = lambda x: np.percentile(x, [25, 75])
        sns.pointplot(errorbar=error_bar_percentile,
                      data=df, x="modules", y=key,
                      join=False,
                      capsize=.30,
                      markers="d",
                      scale=1.0,
                      err_kws={'linewidth': 0.7}, estimator=np.median,
                      color=self.color,
                      label=key.split("_")[0] + " " + " ".join(self.experiment_name.split("_")),
                      ax=ax)

        if show_:
            plt.show()

    def get_and_plot(self, path=None, map_path=None, has_seen_softmax=True, ax=None, num_modules=None):
        if path is None:
            path = self.ramping_exp_path

        self.experiment_name = os.path.basename(path)
        results = self.get_metrics_from_all_files(path, has_seen_softmax=has_seen_softmax)
        key = self.metric

        df = self.get_specific_results(results, key, map_path)
        if num_modules is not None:
            df = df[df['modules'] <= num_modules]
        self.plot_result(df, key, ax=ax)

    def get_df(self, path=None, map_path=None, has_seen_softmax=True):
        if path is None:
            path = self.ramping_exp_path

        results = self.get_metrics_from_all_files(path, has_seen_softmax=has_seen_softmax)
        key = self.metric

        df = self.get_specific_results(results, key, map_path)
        return df

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

    def get_map_metrics(self, map_paths, metrics):
        map_results = {}
        for dataset, map_path in map_paths:
            map_results[dataset] = {}
            for metric in metrics:
                metric_val = self.include_map(map_path)
                map_results[dataset][metric] = round(np.median(metric_val[metric]), ndigits=4)
        return map_results



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
        if len(self.indices) > self.current_index + 1:
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


def get_best_chosen_modules(pcl, num_modules):
    return pcl['module_selection'][num_modules]


def read_pcl(path_to_pcl, num_modules):
    pcl = pickle.load(open(path_to_pcl, 'rb'))
    modules = get_best_chosen_modules(pcl, num_modules)
    return modules


def read_write_all(path_to_runs, save_path, num_modules):
    files = os.listdir(path_to_runs)
    module_names = {}
    for file in files:
        name = file.split(".")[0]
        if 'run' in name:
            run_number = int(name.split("_")[1])
            path = os.path.join(path_to_runs, file)
            path = os.path.join(path, f"run_number_{run_number}.pkl")
            module_names[run_number] = read_pcl(path, num_modules)

    with open(os.path.join(save_path, f'module_names_{num_modules}_modules.pkl'), 'wb') as handle:
        pickle.dump(module_names, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_experiment_to_path_mapping(experiment_path):
    all_paths = os.listdir(experiment_path)
    paths = {}
    for path in all_paths:
        if 'operator_norm_ramping' in path and 'subclass' not in path and 'attn' not in path:
            if 'min' not in path:
                if '_ll' in path:
                    paths['operator_norm_ramping_mlp_ll'] = path
                elif 'acc' in path:
                    paths['operator_norm_ramping_mlp_acc'] = path
                else:
                    paths['operator_norm_ramping_mlp'] = path

            else:
                paths['operator_norm_ramping_mlp_min'] = path

        if 'operator_norm_ramping' in path and ('attn' in path or 'subclass' in path):
            if 'min' not in path:
                if '_ll' in path:
                    paths['operator_norm_ramping_attn_ll'] = path
                elif 'acc' in path:
                    paths['operator_norm_ramping_attn_acc'] = path
                else:
                    paths['operator_norm_ramping_attn'] = path
            else:
                paths['operator_norm_ramping_attn_min'] = path

        if 'random_ramping' in path:
            if 'll' in path:
                paths['last_layer'] = path
            else:
                paths['random_ramping'] = path
        if 'last_layer' in path:
            if 'full' in path:
                paths['last_layer_full'] = path
            else:
                paths['last_layer'] = path

        if 'sublayer_full' in path:
            if 'acc' in path:
                paths['sublayer_full_acc'] = path
            elif 'fine_grained' in path:
                paths['sublayer_full_fine_grained'] = path
            else:
                paths['sublayer_full'] = path
        elif 'sublayer_predefined' in path:
            paths['sublayer_predefined'] = path
        elif 'sublayer' in path:
            if 'acc' in path:
                paths['sublayer_full_acc'] = path
            else:
                paths['sublayer_full'] = path

    new_dict = {key: os.path.join(experiment_path, val) for key, val in paths.items()}
    return new_dict


def make_laplace_plot_one(experiment_path, map_path=None, save_path="", metric = 'nll', val_path = ""):
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)
    names = ['Operator norm MLP', 'Operator norm attn', 'Random ramping']
    keys = ['operator_norm_ramping_mlp', 'operator_norm_ramping_attn', 'random_ramping']
    experiment_paths = [experiment_to_paths[key] for key in keys]

    last_layer_path = experiment_to_paths['last_layer_full'] if 'last_layer_full' in experiment_to_paths.keys() else experiment_to_paths['last_layer']

    last_layer_name = 'Last layer'

    mp_path = None if val_path else map_path

    plotter = MultipleRampingExperiments(experiment_paths, names, mp_path,metric = metric, method='Laplace')
    fig, ax = plt.subplots(1, 1)
    plotter.plot_all(fig=fig, ax=ax, subset_=13)
    # plotter.draw_line_at_best(last_layer_path, ax, last_layer_name, color='tab:green', best_mod=False)
    # last_layer_path = experiment_to_paths['last_layer']
    # last_layer_name = 'Last layer KFAC'
    # plotter.draw_line_at_best(last_layer_path, ax, last_layer_name, color='tab:brown', best_mod=False)
    new_labels = []
    lines = []
    for line, label in zip(*ax.get_legend_handles_labels()):
        if 'mlp' in label.lower():
            new_labels.append('Max $||T||$ MLP')
        if 'attn' in label.lower():
            new_labels.append('Max $||T||$ attn.')
        if 'random' in label.lower():
            new_labels.append('Random')
        lines.append(line)

    ticks = ax.get_xmajorticklabels()
    for tick in ticks:
        text = tick.get_text()
        tick.set_text(text.split(".")[0])
    ax.set_xticklabels(ticks)
    ax.legend(lines, new_labels, loc='upper center', bbox_to_anchor=(0.5, 1.01),
              ncol=1, fancybox=True, shadow=True)
    ax.set_title('Ramping Experiments Laplace SST-2')
    if val_path and map_path is not None:
        nlls, eces = include_temperature_scaling(plotter, map_path, val_path)
        best_score = np.median(nlls) if metric == 'nll' else np.median(eces)
        ax.axhline(y=best_score, linestyle='--', linewidth=2.5, alpha=0.7,
                   color='tab:green', label='Temp. Scaled MAP')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                  ncol=1, fancybox=True, shadow=True)

    fig.tight_layout()
    if save_path:
        if map_path is not None:
            save_name = f"laplace_plot_one_full_{metric}_w_map.pdf"
            fig.savefig(os.path.join(save_path, save_name))
        else:
            save_name = f"laplace_plot_one_full_{metric}.pdf"
            fig.savefig(os.path.join(save_path, save_name))

    plt.show()


def make_laplace_plot_two(experiment_path, map_path=None, save_path=""):
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)
    names_mlp = ['Max operator norm MLP', 'Min operator norm MLP']
    names_attn = ['Max operator norm attn.', 'Min operator norm attn']
    keys_mlp = ['operator_norm_ramping_mlp', 'operator_norm_ramping_mlp_min']
    keys_attn = ['operator_norm_ramping_attn', 'operator_norm_ramping_attn_min']

    exp_paths_mlp = [experiment_to_paths[key] for key in keys_mlp]
    exp_paths_attn = [experiment_to_paths[key] for key in keys_attn]

    plotter_mlp = MultipleRampingExperiments(exp_paths_mlp, names_mlp, map_path, method='Laplace')
    plotter_attn = MultipleRampingExperiments(exp_paths_attn, names_attn, map_path, method='Laplace')

    fig, ax = plt.subplots(1, 1)
    plotter_mlp.plot_all(fig=fig, ax=ax)
    fig.tight_layout()
    if save_path:
        if map_path is not None:
            fig.savefig(os.path.join(save_path, 'laplace_plot_two_mlp_w_map.pdf'))
        else:
            fig.savefig(os.path.join(save_path, 'laplace_plot_two_mlp.pdf'))

    plt.show()
    fig, ax = plt.subplots(1, 1)
    plotter_attn.plot_all(fig=fig, ax=ax)
    fig.tight_layout()
    if save_path:
        if map_path is not None:
            fig.savefig(os.path.join(save_path, 'laplace_plot_two_attn_w_map.pdf'))
        else:
            fig.savefig(os.path.join(save_path, 'laplace_plot_two_attn.pdf'))

    plt.show()


def choose_right_ones_full(path):
    # experiment_to_paths = make_experiment_to_path_mapping(experiment_path)
    if 'RTE' in path:
        names = ['Random ramping', 'Last layer F. GGN'][::-1]
        keys = ['operator_norm_ramping_mlp', 'last_layer'][::-1]

    elif 'sst2_nll' in path:
        names = ['Random ramping', 'Last layer'][::-1]
        keys = ['operator_norm_ramping_mlp', 'last_layer'][::-1]

    elif 'SST2' in path:
        names = ['Random ramping', 'Last layer F. GGN'][::-1]
        keys = ['operator_norm_ramping_mlp', 'last_layer'][::-1]

    elif 'MRPC' in path:
        names = ['Random ramping', 'Last layer F. GGN'][::-1]
        keys = ['operator_norm_ramping_mlp', 'last_layer'][::-1]

    elif 'imdb' in path:
        names = ['Operator norm attn', 'Last layer']
        keys = ['operator_norm_ramping_attn', 'last_layer']
    else:
        raise ValueError("Could not decipher path")
    return names, keys


def choose_right_ones_speficic(path):
    # experiment_to_paths = make_experiment_to_path_mapping(experiment_path)

    if 'RTE' in path:
        names = ['Max operator norm attn. + LL', 'S-KFAC full model', 'Last layer']
        keys = ['operator_norm_ramping_attn_ll', 'sublayer_full', 'last_layer']

    elif 'sst2_nll' in path:
        names = ['Max operator norm MLP', 'S-KFAC full model', 'Last layer']
        keys = ['operator_norm_ramping_mlp', 'sublayer_full', 'last_layer']

    elif 'SST2' in path:
        names = ['Max operator norm MLP', 'S-KFAC full model', 'Last layer']
        keys = ['operator_norm_ramping_mlp', 'sublayer_full', 'last_layer']

    elif 'MRPC' in path:
        names = ['Max operator norm MLP', 'S-KFAC full model', 'Last layer']
        keys = ['operator_norm_ramping_mlp', 'sublayer_full', 'last_layer']

    elif 'imdb' in path:
        names = ['Max operator norm MLP', 'S-KFAC full model', 'Last layer']
        keys = ['operator_norm_ramping_mlp', 'sublayer_full', 'last_layer']

    else:
        raise ValueError("Could not decipher path")
    return names, keys

def include_temperature_scaling(plotter, map_path, val_path):

    runs = [0,1,2,3,4]
    map_paths, val_paths = [],[]
    for run in runs:
        map_paths.append(os.path.join(map_path, f"run_number_{run}_map.pkl"))
        val_paths.append(os.path.join(val_path, f"run_number_{run}_map.pkl"))

    nlls, eces = [], []
    for mp, vp in zip(map_paths, val_paths):
        nll, ece = plotter.calc_temperature_scaling(mp, vp)
        nlls.append(nll)
        eces.append(ece)

    return nlls, eces

def make_laplace_plot_three_full(experiment_path, map_path=None, save_path="", metric = 'nll', val_path = None,
                                 new_ = True, ax = None, subset = None):
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)

    names = ['S-KFAC full model']
    keys = ['sublayer_full_fine_grained']

    exp_paths = [experiment_to_paths[key] for key in keys]

    colors = ['tab:green', 'tab:orange', 'tab:brown']
    names_, keys_ = choose_right_ones_full(experiment_path)
    #
    # plotter = MultipleRampingExperiments(exp_paths, names, map_path if val_path is None else None,
    #                                      sublayer_ramping=True, metric = metric, method='Laplace')
    plotter = MultipleRampingExperiments(exp_paths, names,map_path=None,
                                         sublayer_ramping=False, metric = metric, method='Laplace')

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.ticklabel_format(axis='x', style='sci', useMathText=True)
    df = plotter.plot_all(ax=ax, use_num_floats=False, return_the_effing_dataframe=True)
    if subset == 2:
        df = df[df['modules'] != 1]
        df = df[df['modules'] > 3.4]
    elif subset == 0:
        df = df[df['modules'] > 1.2]

    markersize = 20
    capsize = 16
    elinewidth = 3
    capthick = 3
    best_scores = []
    best_mods = []
    scores = []
    num_modules = [1,1]
    for idx, (n, k) in enumerate(zip(names_, keys_)):
        p = experiment_to_paths[k]
        col = colors[idx]
        bs, bm, sc = plotter.draw_line_at_best(p, ax, name=n, color=col,num_modules=num_modules[idx], set_point_instead=True,
                                                      return_stuff=True)
        best_scores.append(bs)
        best_mods.append(bm)
        scores.append(sc)

    scores = [list(sc) for sc in scores]
    percentage_to_results = {}
    for mod in df['modules'].unique():
        if mod not in percentage_to_results:
            percentage_to_results[mod] = list(df[df['modules'] == mod]['nll'])

    xvals = []
    for key in percentage_to_results.keys():
        memory_calculator = MemoryCalculator(percentage = key, laplace = True)
        xvals.append(memory_calculator.memory_calculator())

    yvals = []
    label = 'S-KFAC full model'
    ranges = []
    for idx, (key, val) in enumerate(percentage_to_results.items()):
        yvals.append(np.median(val))
        ranges.append(yvals[-1] - np.percentile(val, [25, 75]))

    argmax_ = np.argmax(xvals)
    max_x, max_y, max_range = xvals[argmax_], yvals[argmax_], ranges[argmax_]
    xvals = [x for idx, x in enumerate(xvals) if idx != argmax_]
    yvals = [x for idx, x in enumerate(yvals) if idx != argmax_]
    ranges = [x for idx, x in enumerate(ranges) if idx != argmax_]
    ax.plot(xvals, yvals, 'd', color = 'tab:red', markersize = markersize, label = label)
    ax.errorbar(xvals, yvals, yerr = np.abs(np.array(ranges).T), color = 'tab:red',
                    elinewidth = elinewidth, capsize = capsize, fmt = 'none', capthick = capthick)

    ax.plot(max_x, max_y, 'd', color = 'tab:orange', markersize = markersize, label = 'Fully Stochastic')
    ax.errorbar(max_x, max_y, yerr=np.abs(np.array(max_range)[:, None]), color = 'tab:orange',
                elinewidth = elinewidth, capsize = capsize, fmt = 'none', capthick = capthick)

    num_floats = (768*2)**2 + (768*2)
    error = np.abs(np.array(np.median(scores[0]) - np.percentile(scores[0], [25, 75]))[:, None])
    ax.plot(num_floats, best_scores[0], 'd', color='tab:pink', markersize=markersize, label='Last Layer LA F. GGN')
    ax.errorbar(num_floats, best_scores[0], yerr = error, color = 'tab:pink',
                    elinewidth = elinewidth, capsize = capsize, fmt = 'none', capthick = capthick)

    memory_calculator = MemoryCalculator(num_modules=1, module_type='mlp', laplace = True)
    num_floats = memory_calculator.memory_calculator()
    error = np.abs(np.array(np.median(scores[1]) - np.percentile(scores[1], [25, 75]))[:, None])
    ax.plot(num_floats, best_scores[1], 'd', color='tab:blue', markersize=markersize, label='Max $||T||$ MLP 1 Module')
    ax.errorbar(num_floats, best_scores[1], yerr = error, color = 'tab:blue',
                    elinewidth = elinewidth, capsize = capsize, fmt = 'none', capthick = capthick)

    ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel('Num. Floats')
    ax.set_ylabel('NLL')
    # ax.set_title('Laplace RTE')
    def str_num_sci_str_num(num):
        num = float(num)
        ugly = "{:.1e}".format(num).split("e")
        pm = ugly[-1][0]
        number = ugly[-1][1:]
        if number[0] == '0':
            number = number[-1]
        new_number = "$" + ugly[0] +"\cdot " +  "10^{" + f"{number}" + "}$"
        return new_number

    # ax.set_xscale('log')
    # tick_labels = ax.get_xmajorticklabels()
    # new_ticks = []
    # counter = 0
    # for idx, tick in enumerate(tick_labels):
    #     if (idx +1)  %  3 == 0:
    #         ugly_number = tick.get_text()
    #         scientific_number = str_num_sci_str_num(ugly_number)
    #         tick.set_text(scientific_number)
    #         new_ticks.append(tick)
    #         continue
    #     tick.set_text("")
    #     new_ticks.append(tick)
    # ax.set_xticklabels(new_ticks)



    if val_path is not None and map_path is not None:
        nlls, eces = include_temperature_scaling(plotter, map_path, val_path)
        best_score = np.median(nlls) if metric == 'nll' else np.median(eces)
        ybounds = ax.get_ybound()
        diff = ybounds[1] - ybounds[0]

        if best_score > ybounds[1]:
            new_lim = best_score + diff / 5
            ax.set_ybound(ybounds[0], new_lim)

        elif ybounds[1] < (best_score - diff / 4):
            y1 = best_score + diff / 3
            ax.set_ybound(ybounds[0], y1)
        # min_ = 100
        # for line in ax.get_lines():
        #     data = line.get_data()[-1]
        #     if np.min(data) < min_:
        #         min_ = np.min(data)

        # ax.set_ybound(min_ - 1e-3,ybounds[1])
        color = 'tab:red'
        label_name = 'Temp. scaled MAP'

        if not new_:
            tick_labels = ax.get_xmajorticklabels()
            for tick in tick_labels:
                if int(tick._text[0]) >= 1:
                    if len(tick._text) > 3:
                        tick.set_text(tick._text[:-1])

            if tick_labels[-1]._text == '100.':
                tick_labels[-1].set_text('100')
                # ax.set_xticklabels(tick_labels)
            ax.set_xticklabels(tick_labels)
        ax.axhline(y=best_score, linestyle='--', linewidth=4, alpha=0.7,
                   color='tab:red' if color is None else color, label=label_name)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
    #           ncol=1, fancybox=True, shadow=True)
        # ax.set_xscale('log')
        # ax.xaxis.set_major_formatter('{x:1.3f}')

    # fig.tight_layout()
    if save_path:
        if map_path is not None and val_path is None:
            save_name = f"laplace_plot_three_full_{metric}_w_map.pdf"
            fig.savefig(os.path.join(save_path, save_name))
        else:
            save_name = f"laplace_plot_three_RTE_full_fgrained_{metric}.pdf"
            fig.savefig(os.path.join(save_path, save_name))

    return ax
    # plt.show()


def make_laplace_plot_three_predefined(experiment_path, map_path=None, save_path=""):
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)

    names = ['S-KFAC predifined modules']
    keys = ['sublayer_predefined']

    exp_paths = [experiment_to_paths[key] for key in keys]

    colors = ['tab:green', 'tab:orange', 'tab:brown']
    names_, keys_ = choose_right_ones_speficic(experiment_path)
    best_num_modules = 2
    plotter = MultipleRampingExperiments(exp_paths, names, map_path, sublayer_ramping=True, method='Laplace')

    fig, ax = plt.subplots(1, 1)
    plotter.plot_all(fig=fig, ax=ax)
    plotter.draw_line_at_best(experiment_to_paths[keys_[0]], ax, names_[0], color='tab:blue', num_modules=2,
                              set_point_instead=True)
    for idx, (n, k) in enumerate(zip(names_[1:], keys_[1:])):
        p = experiment_to_paths[k]
        col = colors[idx]
        sublayer_ramping = True if 'sublayer' in k else False
        plotter.draw_line_at_best(p, ax, name=n, color=col, sublayer_ramping=sublayer_ramping)

    fig.tight_layout()
    if save_path:
        if map_path is not None:
            fig.savefig(os.path.join(save_path, 'laplace_plot_three_predefined_w_map.pdf'))
        else:
            fig.savefig(os.path.join(save_path, 'laplace_plot_three_predefined.pdf'))

    plt.show()


def make_rte_laplace_ll_plot(experiment_path, map_path = None, save_path = ""):
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)
    names = ['Operator norm MLP + LLLA', 'Operator norm attn + LLLA']
    keys = ['operator_norm_ramping_mlp_ll', 'operator_norm_ramping_attn_ll']
    experiment_paths = [experiment_to_paths[key] for key in keys]

    last_layer_path = experiment_to_paths['last_layer_full']
    last_layer_name = 'Last layer full'
    random_ramping_path = experiment_to_paths['random_ramping']
    random_ramping_name = 'Random ramping'

    plotter = MultipleRampingExperiments(experiment_paths, names, map_path, method='Laplace')
    fig, ax = plt.subplots(1, 1)
    plotter.plot_all(fig=fig, ax=ax)
    plotter.draw_line_at_best(last_layer_path, ax, last_layer_name, color='tab:green', best_mod=False)
    plotter.draw_line_at_best(random_ramping_path, ax, random_ramping_name, color = 'tab:brown')
    fig.tight_layout()
    if save_path:
        if map_path is not None:
            fig.savefig(os.path.join(save_path, 'laplace_plot_ll_rte_w_map.pdf'))
        else:
            fig.savefig(os.path.join(save_path, 'laplace_plot_ll_rte.pdf'))

    plt.show()
def make_plot_one_swag(experiment_path, map_path=None, save_path=""):
    # experiment_to_paths = make_experiment_to_path_mapping(experiment_path)

    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)
    names = [ 'Operator norm attn','Operator norm MLP', 'Random ramping']
    keys = ['operator_norm_ramping_attn', 'operator_norm_ramping_mlp',  'random_ramping']
    experiment_paths = [experiment_to_paths[key] for key in keys]

    plotter = MultipleRampingExperiments(experiment_paths, names, map_path, method='SWAG')
    fig, ax = plt.subplots(1, 1)
    plotter.plot_all(fig=fig, ax=ax)
    # plotter.draw_line_at_best(last_layer_path, ax, last_layer_name, color='tab:green', best_mod=False)
    fig.tight_layout()
    if save_path:
        if map_path is not None:
            fig.savefig(os.path.join(save_path, 'swag_plot_one_nll_w_map.pdf'))
        else:
            fig.savefig(os.path.join(save_path, 'swag_plot_one_nll.pdf'))
    plt.show()
    plotter = MultipleRampingExperiments(experiment_paths, names, map_path, metric='accuracy_score', method='SWAG')
    fig, ax = plt.subplots(1, 1)

    plotter.plot_all(fig=fig, ax=ax)
    fig.tight_layout()
    if save_path:
        if map_path is not None:
            fig.savefig(os.path.join(save_path, 'swag_plot_one_acc_w_map.pdf'))
        else:
            fig.savefig(os.path.join(save_path, 'swag_plot_one_acc.pdf'))
    plt.show()


def make_plot_two_full_swag(experiment_path, map_path=None, save_path=""):
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)

    names = ['Sublayer full model']
    keys = ['sublayer_full']
    full_path = experiment_to_paths['random_ramping']
    full_name = ""
    experiment_paths = [experiment_to_paths[key] for key in keys]
    other_names = ['Operator norm MLP', 'Operator norm attn', 'Random ramping']
    other_keys = ['operator_norm_ramping_mlp', 'operator_norm_ramping_attn', 'random_ramping']

    colors = ['tab:green', 'tab:orange', 'tab:brown']
    plotter = MultipleRampingExperiments(experiment_paths, names, map_path, method='SWAG')
    fig, ax = plt.subplots(1, 1)
    plotter.plot_all(fig=fig, ax=ax)
    for idx, (key, name) in enumerate(zip(other_keys, other_names)):
        plotter.draw_line_at_best(experiment_to_paths[key], ax=ax, name=name, color=colors[idx])
    plotter.draw_line_at_best(full_path, ax=ax, name=full_name, num_modules=38, set_point_instead=True)
    ax.set_xlabel('Percentages')
    # ax.set_ybound(0.5, 2.5)
    fig.tight_layout()
    if save_path:
        if map_path is not None:
            fig.savefig(os.path.join(save_path, 'swag_plot_two_nll_w_map.pdf'))
        else:
            fig.savefig(os.path.join(save_path, 'swag_plot_two_nll.pdf'))
    plt.show()

    plotter = MultipleRampingExperiments(experiment_paths, names, map_path, metric='accuracy_score', method='SWAG')
    fig, ax = plt.subplots(1, 1)
    plotter.plot_all(fig=fig, ax=ax)

    plotter.draw_line_at_best(full_path, ax = ax, name = full_name, num_modules=38, set_point_instead=True)

    for idx, (key, name) in enumerate(zip(other_keys, other_names)):
        plotter.draw_line_at_best(experiment_to_paths[key], ax=ax, name=name, color=colors[idx])


    if 'mrpc' in experiment_path.lower():
        ax.set_ybound(0.76, 0.95)
    ax.set_xlabel('Percentages')
    fig.tight_layout()
    if save_path:
        if map_path is not None:
            fig.savefig(os.path.join(save_path, 'swag_plot_two_acc_w_map.pdf'))
        else:
            fig.savefig(os.path.join(save_path, 'swag_plot_two_acc.pdf'))

    plt.show()



def find_diff_between_map_and_other(map_path, other_path, data_path, save_path = ""):

    data = pd.read_csv(data_path)
    map_eval = pickle.load(open(map_path, 'rb'))
    other_eval = pickle.load(open(other_path, 'rb'))
    map_eval = map_eval['results']
    softmax_ = nn.Softmax(dim = 1)
    best = other_eval['results']['1.0']
    labels_eval = best.labels
    labels_map = map_eval.labels
    map_preds, best_preds = softmax_(map_eval.predictions).numpy(), best.predictions.numpy()
    labels = labels_map.numpy()
    labels_data = np.asarray(data['label'])
    assert np.sum(labels - labels_data) == 0

    _,data_cls_one, diff_one = find_stuff_for_class(labels, map_preds, best_preds, 0, np.array([data['sentence1'], data['sentence2']]))
    _,data_cls_two, diff_two = find_stuff_for_class(labels, map_preds, best_preds, 1, np.array([data['sentence1'], data['sentence2']]))
    data_one = pd.DataFrame(data = data_cls_one)
    data_two = pd.DataFrame(data = data_cls_two)
    data_one['sentences'] = data_cls_one
    data_two['sentences'] = data_cls_two
    if save_path:
        data_one.to_csv(os.path.join(save_path, 'wrongly_labeled_data_cls_one.txt'), index = False)
        data_two.to_csv(os.path.join(save_path, 'wrongly_labeled_data_cls_two.txt'), index = False)


def find_diff_between_map_and_other_swag(map_path, other_path, data_path):
    data = pd.read_csv(data_path)
    map_eval = pickle.load(open(map_path, 'rb'))
    other_eval = pickle.load(open(other_path, 'rb'))
    map_eval = map_eval['results']
    softmax_ = nn.Softmax(dim=1)
    best = other_eval[5]
    labels_map = map_eval.labels
    map_preds, best_preds = softmax_(map_eval.predictions).numpy(), best.predictions.numpy()
    labels = labels_map.numpy()
    labels_data = np.asarray(data['label'])
    assert np.sum(labels - labels_data) == 0

    data_cls = find_stuff_for_class_swag(labels, map_preds, best_preds, 0, np.array([data['sentence1'], data['sentence2']]))
    return data_cls
def find_stuff_for_class_swag(labels, preds_map, preds_swa, class_number, data):

    data = data.T
    correct_other = list(np.argwhere(preds_swa.argmax(-1) == labels).flatten())
    correct_map = list(np.argwhere(preds_map.argmax(-1) == labels).flatten())
    disjoint =  list(set(correct_other) - set(correct_map))
    disjoint_labels = labels[disjoint]
    highest_scoring = np.argsort(preds_swa[disjoint, disjoint_labels])[::-1]
    data_cls = data[disjoint][highest_scoring]
    return data_cls




def find_stuff_for_class(labels, preds_map, preds_la, class_number, data):

    data = data.T
    class_indices = labels == class_number
    pred_ma, pred_la, data_cls = preds_map[class_indices], preds_la[class_indices], data[class_indices]
    wrongly_predicted = np.argwhere(pred_ma.argmax(-1) != class_number)
    pred_ma = pred_ma[wrongly_predicted.squeeze(1)]
    pred_la = pred_la[wrongly_predicted.squeeze(1)]
    data_cls = data_cls[wrongly_predicted.squeeze(1)]
    diffs = pred_la[:, class_number] - pred_ma[:, class_number]
    best_ones = np.argsort(diffs)[::-1]
    diffs = diffs[best_ones]
    percentages = diffs/np.sum(diffs)*100
    data_cls = data_cls[best_ones]
    repeated_sentences = []
    for perc, sentence in zip(percentages, data_cls):
        num_rep = int(perc * 30)
        if num_rep == 0:
            num_rep = 1
        repeated_sentences += [sentence]*num_rep

    return repeated_sentences, data_cls, diffs


def make_main_plot_specific_experiment(ax, experiment_path, counter, metric = 'nll'):
    """

    Hello Gustav, welcome to the madness.. this should give you a helping hand
    Also this is a helper function, the main function is juuuust below
    :param ax: Axes object meant for plotting in fig, ax = plt.subplots(2, 3) for ECE below and NLL above,
                This is supposed to be an indexed ax object i.e ax[i,j] so functions like ax.plot() are accessible
    :param experiment_path: Path to experiments, either Laplace or SWAG paths, names are read automatically
    :param counter: Counter to how many experiments have already been plotted so they aren't positioned on top of
                    eachother
    :return: Updated ax object, and counter + number of newly plotted experiments: I.e the function is designed for
            an outer loop
    """

    if 'laplace' in experiment_path.lower():
        experiment_name_prefix = 'LA'
        keys = ['operator_norm_ramping_mlp', 'operator_norm_ramping_attn', 'random_ramping', 'sublayer_full_fine_grained',
                'last_layer_full']
        names = ['Max $||T||$ MLP LA', 'Max $||T||$ attn. LA', 'Random LA', 'S-KFAC LA',
                 'Last Layer LA F. GGN']
        point_types = ['o', 'v', 'p', 's','d']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']
    elif 'swag' in experiment_path.lower():
        experiment_name_prefix = 'SWAG'
        keys = ['operator_norm_ramping_mlp', 'operator_norm_ramping_attn', 'random_ramping', 'sublayer_full']
        names = ['Operator norm SWAG', 'Operator norm attn. SWAG', 'Random ramping SWAG', 'Sublayer l1 SWAG']
        keys = ['random_ramping']
        names = ['Random SWAG']
        point_types = ["P", '*', 'H', 'd']
        colors = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
    else:
        raise ValueError("Either 'laplace' or 'swag' must be in experiment_path.lower() to ascertain which experiments"
                         "we are working on you dumb fuck")

    def plot_point(ax, x_val, y_val, marker, color, label):

        ax.plot(x_val, y_val, marker = marker, color = color, label = label, markersize = 18)
        ylims = ax.get_ybound()
        if y_val > ylims[1]:
            new_ylim = (ylims[0], y_val)
            ax.set_ybound(new_ylim)
        elif y_val < ylims[0]:
            new_ylim = (y_val, ylims[1])
            ax.set_ybound(new_ylim)

        return ax

    def plot_uncertainty(ax, x, y, yvals, point, color, name):

        error = np.array([y- np.percentile(yvals, 25), np.percentile(yvals, 75)-y])[:, None]
        ax.errorbar(x, y, error, fmt = 'None',color = color, elinewidth = 3.8)
        return ax
    def get_experiment_type(name, laplace = True):
        if laplace:
            if 'att' in name.lower():
                return 'attn'
            elif 'mlp' in name.lower():
                return 'mlp'
            elif 'random' in name.lower():
                return 'random'
            elif 'kfac' in name.lower():
                return 'sublayer'
            elif 'last' in name.lower():
                return 'last_layer'
            else:
                breakpoint()
        else:
            return 'random'
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)
    experiment_paths = [experiment_to_paths[key] for key in keys]

    plotter = MultipleRampingExperiments(experiment_paths, metric=metric)
    laplace = 'laplace' in experiment_path.lower()
    swag = 'swag' in experiment_path.lower()

    for path, name, point, color in zip(experiment_paths, names, point_types, colors):
        best_mod = globals().get(f'best_mod_{path}', None)
        experiment_type = get_experiment_type(name, laplace)
        if experiment_type == 'sublayer':
            max_mod = 62
        else:
            max_mod = None
        best_val, best_mod, uncertainty = plotter.find_best_measure(path, best_mod, max_mod, return_uncertainty=True)
        max_, min_ = np.max(uncertainty), np.min(uncertainty)

        if experiment_type == 'sublayer':
            percentage = best_mod
            num_modules = None
        else:
            percentage = None
            num_modules = best_mod

        memory_calculator = MemoryCalculator(num_modules, percentage, experiment_type, laplace, swag)
        memory_usage = memory_calculator.memory_calculator()

        plot_point(ax, memory_usage, best_val, point, color, name)
        plot_uncertainty(ax, memory_usage,best_val, uncertainty, point, color, name)
        counter += 1
        globals().__setitem__(f'best_mod_{path}', best_mod)

    return ax, counter

def update_data_dict(data_dict, path, dataset,metric = 'nll', la = True):


    if la:
        keys = ['operator_norm_ramping_mlp', 'operator_norm_ramping_attn', 'random_ramping',
                'operator_norm_ramping_mlp_min','operator_norm_ramping_attn_min']
        names = ['Max operator norm LA', 'Max operator norm attn. LA', 'Random ramping LA',
                 'Min operator norm LA', 'Min operator norm attn. LA']
        point_types = ['o', 'v', 'p', 's', 'd']

    else:
        keys = ['operator_norm_ramping_mlp', 'operator_norm_ramping_attn', 'random_ramping', 'sublayer_full']
        names = ['Operator norm SWAG', 'Operator norm attn. SWAG', 'Random ramping SWAG', 'Sublayer l1 SWAG']
        keys = ['random_ramping']
        names = ['Random ramping SWAG']


    def get_medians_and_take_out(df, maximum = 11):

        df = df[df['modules'] <= maximum]
        modules = sorted(df['modules'].unique())
        medians = []
        for mod in modules:
            medians.append(df[df['modules'] == mod][metric].median())

        return medians

    experiment_to_paths = make_experiment_to_path_mapping(path)
    experiment_paths = [experiment_to_paths[key] for key in keys]

    plotter = MultipleRampingExperiments(experiment_paths, metric=metric)

    dataframe_dict = plotter.get_dfs()
    for key, df_key in zip(keys, dataframe_dict.keys()):
        dataframe = dataframe_dict[df_key]
        if key not in data_dict:
            data_dict[key] = []
        medians = get_medians_and_take_out(dataframe)
        data_dict[key].append(medians)

    return data_dict

def make_ranking_plot(experiment_paths, save_path = ""):

    def extract_laplace_and_swag(experiment_path):
        if any(('swag' in experiment_path, 'laplace' in experiment_path)):
            raise ValueError("This is a function that is supposed to plot for both experiments you dumb fucker")

        laplace_path = [os.path.join(experiment_path, p)
                        for p in os.listdir(experiment_path) if 'laplace' in p.lower()][0]
        swag_path = [os.path.join(experiment_path, p) for p in os.listdir(experiment_path) if 'swag' in p.lower()][0]
        return laplace_path, swag_path

    dataset_name_to_swag_la_paths = {}
    for path in experiment_paths:
        la_path, swa_path = extract_laplace_and_swag(path)
        if 'sst2' in path.lower():
            dataset_name_to_swag_la_paths['SST2'] = {'LA': la_path, 'swa': swa_path}
        elif 'mrpc' in path.lower():
            dataset_name_to_swag_la_paths['MRPC'] = {'LA': la_path, 'swa': swa_path}
        elif 'rte' in path.lower():
            dataset_name_to_swag_la_paths['RTE'] = {'LA': la_path, 'swa': swa_path}
        else:
            raise ValueError("Why are you giving paths that dont contain the dataset name, are you out of your mind???")

    module_numbers = [1,2,3,4,5,6,7]
    font = {'family': 'serif',
            'size': 20,
            'serif': 'cmr10'
            }
    mpl.rc('font', **font)
    mpl.rc('legend', fontsize=17)
    mpl.rc('axes', labelsize=25)
    # fig, ax = plt.subplots(2, 3, figsize = (13,9.6))   # [NLL x ECE] [SST2 MRPC RTE]^T  if you enjoy some outer products :))
    metrics = ['nll', 'MCE']
    datasets = ['SST2', 'MRPC', 'RTE']

    data_dict = {}
    for j, dataset in enumerate(datasets):
        counter = 0
        la_path, swa_path = dataset_name_to_swag_la_paths[dataset]['LA'], dataset_name_to_swag_la_paths[dataset]['swa']
        data_dict = update_data_dict(data_dict, la_path, dataset)

    def give_score(argsorted):
        scores = np.zeros((5, ))
        for i, idx in enumerate(argsorted):
            scores[idx] += i

        return scores
    scores_counter = np.array([1,2,3,4,5])
    full_rankings = np.zeros((len(module_numbers),len(data_dict)))
    for idx, module_number in enumerate(module_numbers):
        names = list(data_dict.keys())
        vps = []
        for i in range(len(datasets)):
            values = [data_dict[name][i][idx] for name in names]
            argsorted = np.argsort(values)
            vps.append(give_score(argsorted))
        vps = np.array(vps)
        medians = np.median(vps, axis = 0)
        full_rankings[idx] = medians


    names = ['Max $||T||$ MLP', 'Max $||T||$ attn.', 'Random',
             'Min $||T||$ MLP', 'Min $||T||$ attn.']

    plotter = RankingPlotter(full_rankings, names, module_numbers)
    fig, ax = plt.subplots(1,1)
    plotter.draw(ax)
    Line, Label = ax.get_legend_handles_labels()
    # print(Label)
    lines, labels = [], []
    for line in Line:
        line.set_linewidth(0)
    lines.extend(Line)
    labels.extend(Label)
    # labels = [lab[:-3] for lab in labels]
    plt.subplots_adjust(left=0.1,
                        bottom=0.15,
                        right=0.9,
                        top=0.85,
                        wspace=0.4,
                        hspace=0.2)
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncols = 3, fontsize = 14,
               columnspacing=0.8)
    modules_list = [0,1,2,3,4,5,8,11]
    ax.set_xticklabels([str(i) for i in modules_list])
    ax.set_xlabel('Num. Stoch. Modules')
    ax.set_ylabel('Ranking $\downarrow$')

    if save_path:
        fig.savefig(os.path.join(save_path, 'ranking_plot_all.pdf'), format = 'pdf')

    plt.show()

def extract_for_bar_plot(experiment_path, map_path = "", map_val_path = "", metric = ""):

    if 'laplace' in experiment_path.lower():
        experiment_name_prefix = 'LA'
        keys = ['operator_norm_ramping_mlp','operator_norm_ramping_mlp_min', 'operator_norm_ramping_attn',
                'operator_norm_ramping_attn_min',
                'random_ramping', 'sublayer_full_fine_grained',
                'last_layer_full']
        names = ['Max $||T||$ MLP LA','Min $||T||$ MLP LA', 'Max $||T||$ attn. LA',
                 'Min $||T||$ attn. LA','Random LA', 'S-KFAC LA',
                 'Last Layer LA F. GGN']
        point_types = ['o', 'v', 'p', 's','d']
        key_to_max_num_modules = {'operator_norm_ramping_mlp': 2,
                                  'operator_norm_ramping_mlp_min': 2,
                                  'operator_norm_ramping_attn': 8,
                                  'operator_norm_ramping_attn_min': 8,
                                  'random_ramping': 5,
                                  'sublayer_full_fine_grained': 60,
                                  'last_layer_full': 100}
        final_path = 'sublayer_full_fine_grained'
        final_name = 'Fully Stoch. LA'
        final_num_modules = 100
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']
    elif 'swag' in experiment_path.lower():
        experiment_name_prefix = 'SWAG'
        keys = ['operator_norm_ramping_mlp', 'operator_norm_ramping_attn', 'random_ramping', 'sublayer_full']
        names = ['Max $||T||$ MLP SWAG', 'Max $||T||$ attn. SWAG', 'Random SWAG', 'Sublayer $\ell_1$ SWAG']
        point_types = ["P", '*', 'H', 'd']
        colors = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
        key_to_max_num_modules = {'operator_norm_ramping_mlp': 2,
                                  'operator_norm_ramping_attn': 8,
                                  'random_ramping': 5,
                                  'sublayer_full': 11}
        final_path = 'random_ramping'
        final_name = 'Fully Stoch. SWAG'
        final_num_modules = 38
    else:
        raise ValueError("Either 'laplace' or 'swag' must be in experiment_path.lower() to ascertain which experiments"
                         "we are working on you dumb fuck")


    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)
    experiment_paths = [experiment_to_paths[key] for key in keys]
    max_num_modules = [key_to_max_num_modules[key] for key in keys]
    plotter = MultipleRampingExperiments(experiment_paths, metric = metric)
    final_path = experiment_to_paths[final_path]


    exp_names = []
    values = []
    results_holder = {}
    if map_path:
        map_results = list(plotter.ramping_experiments.values())[0].include_map(map_path)
        nll = map_results[metric]
        median_map_nll = np.median(nll)
        results_holder['MAP'] = nll
    if map_val_path and map_path:
        nll, eces = include_temperature_scaling(plotter, map_path, map_val_path)
        if metric != 'nll':
            results_holder['Temp. Scaled MAP'] = eces
        else:
            results_holder['Temp. Scaled MAP'] = nll

    for idx, path in enumerate(experiment_paths):
        best_val, best_mod, uncertainty = plotter.find_best_measure(path,max_mod=max_num_modules[idx],
                                                                    return_uncertainty=True)
        results_holder[names[idx]] = uncertainty

    best_val, best_mod, uncertainty = plotter.find_best_measure(final_path, num_modules=final_num_modules,
                                                                return_uncertainty=True)
    results_holder[final_name] = uncertainty
    return results_holder


def make_bar_plot(experiment_paths, save_path = ""):
    def extract_laplace_and_swag(experiment_path):
        if any(('swag' in experiment_path, 'laplace' in experiment_path)):
            raise ValueError("This is a function that is supposed to plot for both experiments you dumb fucker")

        laplace_path = [os.path.join(experiment_path, p)
                        for p in os.listdir(experiment_path) if 'laplace' in p.lower()][0]
        swag_path = [os.path.join(experiment_path, p) for p in os.listdir(experiment_path) if 'swag' in p.lower()][0]
        map_path = os.path.join(experiment_path, 'map')
        map_val_path = os.path.join(experiment_path, 'map_val')

        return laplace_path, swag_path, map_path, map_val_path

    dataset_name_to_swag_la_paths = {}
    for path in experiment_paths:
        la_path, swa_path, map_path, map_val_path = extract_laplace_and_swag(path)
        if 'sst2' in path.lower():
            dataset_name_to_swag_la_paths['SST2'] = {'LA': la_path, 'swa': swa_path, 'map': map_path, 'map_val':
                                                     map_val_path}
        elif 'mrpc' in path.lower():
            dataset_name_to_swag_la_paths['MRPC'] = {'LA': la_path, 'swa': swa_path, 'map': map_path, 'map_val':
                                                     map_val_path}
        elif 'rte' in path.lower():
            dataset_name_to_swag_la_paths['RTE'] = {'LA': la_path, 'swa': swa_path, 'map': map_path, 'map_val':
                                                     map_val_path}
        else:
            raise ValueError("Why are you giving paths that dont contain the dataset name, are you out of your mind???")

    datasets = ['SST2', 'MRPC', 'RTE']
    dataframe = pd.DataFrame()
    unpack_list = ['LA', 'swa', 'map', 'map_val']
    metrics = ['nll', 'ECE']
    for j, dataset in enumerate(datasets):
        for i, metric in enumerate(metrics):
            la_path, swa_path, map, map_val = [dataset_name_to_swag_la_paths[dataset][up] for up in unpack_list]
            holder = extract_for_bar_plot(la_path, map, map_val, metric = metric)
            experiment_results_holder = holder
            holder = extract_for_bar_plot(swa_path, metric = metric)
            for key, val in holder.items():
                if key not in experiment_results_holder:
                    experiment_results_holder[key] = val

            method_names = []
            results = []
            diff = []
            plus_minus = []
            map_val = np.median(experiment_results_holder['MAP'])

            for key, val in experiment_results_holder.items():
                method_names.append(key)
                results.append(str(np.round(np.median(val), 2)))
                plus_minus.append(str(np.round(np.mean(np.abs(np.median(val) - np.percentile(val, [20,80]))), 3)))
                diff.append(str(np.round(np.median(val) - map_val, 2)))
            if j == 0 and i == 0:
                dataframe['Methods'] = method_names
            dataframe[metric + dataset] = results
            dataframe['Uncertainty'+ f"{i}" +dataset] = plus_minus

    for col in dataframe.columns:
        if 'nll' in col:
            values = [float(val) for val in dataframe[col]]
            argmin = np.argmin(values)
            values = [val if idx != argmin else r"\boldsymbol{" + val + "}" for idx, val in enumerate(dataframe[col]) ]
            dataframe[col] = values

        if 'ECE' in col:
            values = [float(val) for val in dataframe[col]]
            argmin = np.argmin(values)
            values = [val if idx != argmin else r"\boldsymbol{" + val + "}" for idx, val in enumerate(dataframe[col])]
            dataframe[col] = values


    columns = dataframe.columns

    new_dataframe = pd.DataFrame()
    new_dataframe['Methods'] = dataframe['Methods']
    for i, col in enumerate(columns):
        if 'nll' in col or 'ECE' in col:
            pm = dataframe[columns[i+1]]
            values = dataframe[col]
            new_val = []
            for val, p in zip(values, pm):
                new_val.append("$" + val + r"\pm" + p + "$")
            new_dataframe[col] = new_val
            if 'ECE' in col:
                new_dataframe[f'mid{i}'] = ['?']*len(new_dataframe)

    print(new_dataframe.to_latex())

    breakpoint()

def make_main_plot(experiment_paths, save_path = "", lazy = False):

    """
    Hello again Mr Gustav, I am now on my second beer, so the voices are finally starting to quite down
    This is the main function for plotting the things we talked about so have fun
    :param experiment_paths: list[str, str,...]  containing the overall experiment paths for SST2, RTE, and MRPC
            They need not be ordered in that way. It is assumed that both SWAG and Laplace experiments are in subfolders
            to each of those paths for the respective experiments
    :param save_path: str/os.path_like path to the directory you would like to save the figure: file will be saved as pdf
    :return: TBD
    """

    def extract_laplace_and_swag(experiment_path):
        if any(('swag' in experiment_path, 'laplace' in experiment_path)):
            raise ValueError("This is a function that is supposed to plot for both experiments you dumb fucker")

        laplace_path = [os.path.join(experiment_path, p)
                        for p in os.listdir(experiment_path) if 'laplace' in p.lower()][0]
        swag_path = [os.path.join(experiment_path, p) for p in os.listdir(experiment_path) if 'swag' in p.lower()][0]
        return laplace_path, swag_path

    dataset_name_to_swag_la_paths = {}
    for path in experiment_paths:
        la_path, swa_path = extract_laplace_and_swag(path)
        if 'sst2' in path.lower():
            dataset_name_to_swag_la_paths['SST2'] = {'LA': la_path, 'swa': swa_path}
        elif 'mrpc' in path.lower():
            dataset_name_to_swag_la_paths['MRPC'] = {'LA': la_path, 'swa': swa_path}
        elif 'rte' in path.lower():
            dataset_name_to_swag_la_paths['RTE'] = {'LA': la_path, 'swa': swa_path}
        else:
            raise ValueError("Why are you giving paths that dont contain the dataset name, are you out of your mind???")

    def order_stuff_in_plot(ax, metric, column, row, dataset):
        """
        Helper function to make the ax[i,j] behave themselves
        """


        if metric == 'nll':
            y_label = 'NLL'
        elif metric == 'ECE':
            y_label = 'ECE'
        elif metric == 'MCE':
            y_label = 'MCE'
        else:
            y_label = metric
        if column == 0:
            ax.set_ylabel(y_label + r"$\downarrow$")

        ax.yaxis.set_major_formatter('{x:1.3f}')
        ax.set_xscale('log')
        # if row == 1:
        #     ax.set_xlabel(dataset, labelpad=25)
        #     ax.set_xticks([])
        # else:
        #     ax.get_xaxis().set_visible(False)  # there is no reason to have an unordered axis
        ax.grid(True)
        # plt.setp(ax.get_xticklabels(), visible = True)
        # ax.set_xticklabels(ax.get_xmajorticklabels())
        if row == 0:

            if dataset == 'SST2':
                ax.set_title('SST-2', fontsize = 22)
            else:
                ax.set_title(dataset, fontsize = 22)
            # ax.get_xaxis().set_visible(False)
        if row == 1:
            ax.set_xlabel(f'Num. Floats $\downarrow$', labelpad=5)
        return ax

    font = {'family': 'serif',
            'size': 20,
            'serif': 'cmr10'
            }
    mpl.rc('font', **font)
    mpl.rc('legend', fontsize=17)
    mpl.rc('axes', labelsize=25)


    fig, ax = plt.subplots(2, 3, figsize = (13,9.6))   # [NLL x ECE] [SST2 MRPC RTE]^T  if you enjoy some outer products :))
    metrics = ['nll', 'MCE']
    datasets = ['SST2', 'MRPC', 'RTE']

    """
    I think we need larger markersizes, that is changed in make_main_plot_specific_experiment()
    """

    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            counter = 0
            la_path, swa_path = dataset_name_to_swag_la_paths[dataset]['LA'], dataset_name_to_swag_la_paths[dataset]['swa']
            ax[i, j], counter = make_main_plot_specific_experiment(ax[i, j], la_path, counter, metric)
            ax[i, j], counter = make_main_plot_specific_experiment(ax[i, j], swa_path, counter, metric)
            ax[i, j] = order_stuff_in_plot(ax[i,j], metric, j, i, dataset)

    """
    
    The following numbers need to be adjusted to make sure numbers are visible
    """
    plt.subplots_adjust(left=0.1,
                        bottom=0.08,
                        right=0.9,
                        top=0.82,
                        wspace=0.4,
                        hspace=0.2)

    lines = []
    labels = []
    for ax in fig.axes:
        Line, Label = ax.get_legend_handles_labels()
        # print(Label)
        for line in Line:
            line.set_linewidth(0)
        lines.extend(Line)
        labels.extend(Label)
        break


    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncols = 3, fontsize = 22)


    if save_path:
        if os.path.exists(save_path):
            fig.savefig(os.path.join(save_path, 'swag_la_comparison_plot_all_datasets.pdf'))
        else:
            raise ValueError("You fucked up again, and provided a save path that doesn't even exist on your computer")

    plt.show()


def find_and_dataset_paths_and_plot(overall_path, save_path = "", type = 'full'):

    """
    This is a function primarily for me because it really requires that you have put all your stuff
    in the same order as me
    :param overall_path:
    :return:
    """

    names = ['MRPC', 'SST2', 'RTE']
    paths = [os.path.join(overall_path, name) for name in names]
    if type == 'full':
        make_main_plot(experiment_paths=paths, save_path=save_path)
    elif type == 'ranking':
        make_ranking_plot(experiment_paths=paths, save_path=save_path)
    elif type == 'bar':
        make_bar_plot(experiment_paths=paths, save_path=save_path)
    else:
        raise NotImplementedError("")


def make_acc_swag_plots_across_datasets(experiment_paths, map_paths, save_path = ""):

    rte_path = experiment_paths[0]
    ss2_path = experiment_paths[1]
    mrpc_path = experiment_paths[2]

    rte_map = map_paths[0]
    sst2_map = map_paths[1]
    mrpc_map = map_paths[2]

    experiment_to_paths_rte = make_experiment_to_path_mapping(rte_path)
    experiment_to_paths_sst2 = make_experiment_to_path_mapping(ss2_path)
    experiment_to_paths_mrpc = make_experiment_to_path_mapping(mrpc_path)

    names = ['Max operator norm MLP RTE','Max operator norm attn. SST2','Max operator norm MLP MRPC']
    keys = ['operator_norm_ramping_mlp_acc', 'operator_norm_ramping_attn_acc', 'operator_norm_ramping_mlp_acc']

    exp_paths = [experiment_to_paths_rte[keys[0]],experiment_to_paths_sst2[keys[1]],experiment_to_paths_mrpc[keys[2]]]
    plotter = MultipleRampingExperiments(exp_paths, names, [rte_map, sst2_map, mrpc_map], metric = 'nll', method='SWAG')
    fig, ax = plt.subplots(1,1)
    plotter.plot_all(fig, ax)
    ax.set_title('SWAG w. accuracy')
    fig.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, 'swag_across_datasets_acc_trained_NLL.pdf'), format = 'pdf')
    plt.show()
    plotter = MultipleRampingExperiments(exp_paths, names, [rte_map, sst2_map, mrpc_map], metric='accuracy_score', method='SWAG')
    fig, ax = plt.subplots(1, 1)
    plotter.plot_all(fig, ax)
    ax.set_title('SWAG w. accuracy')
    fig.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, 'swag_across_datasets_acc_trained_accuracy.pdf'), format='pdf')
    plt.show()

    breakpoint()


def write_ECE_plot_for_dataset(experiment_path, map_path, metric = 'ECE'):

    last_p_name = os.path.basename(experiment_path)
    if 'laplace' in last_p_name.lower() or 'swag' in last_p_name.lower():
        experiment_path = os.path.dirname(experiment_path)

    add_paths = os.listdir(experiment_path)
    laplace_path, swag_path = experiment_path, experiment_path
    for p in add_paths:
        if 'laplace' in p.lower():
            laplace_path = os.path.join(laplace_path, p)
        if 'swag' in p.lower():
            swag_path = os.path.join(swag_path, p)

    laplace_paths = [os.path.join(laplace_path,p) for p in os.listdir(laplace_path) if 'sublayer' not in p.lower()]
    swag_paths = [os.path.join(swag_path, p) for p in os.listdir(swag_path) if 'sublayer' not in p.lower()]

    plotter_laplace = MultipleRampingExperiments(laplace_paths, map_path=map_path,metric=metric)
    plotter_swag = MultipleRampingExperiments(swag_paths, map_path=map_path, metric=metric)
    plotter_laplace.write_latex_table(bold_direction='method')
    plotter_swag.write_latex_table(bold_direction='method')


def write_map_metrics():

    acc_map_paths = [("SST2", r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\sst2\map"),
                     ("MRPC", r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\mrpc\mrpc_map_acc_100"),
                     ("RTE", r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\rte\map"),
                     ("IMDb", r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb_map_acc_100")]

    nll_map_paths = [("SST2", r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\sst2_nll\map"),
                     ("MRPC", r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\mrpc\map_nll"),
                     ("RTE", r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\rte\map_nll"),
                     ("IMDb", r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\map")]
    metrics = ['nll', 'accuracy_score', 'ECE']
    plotter = RampingExperiments("")
    acc_map_metrics = plotter.get_map_metrics(map_paths=acc_map_paths, metrics=metrics)
    nll_map_metrics = plotter.get_map_metrics(map_paths=nll_map_paths, metrics=metrics)
    print("Accuracy map metrics: ", acc_map_metrics,"\n"
          "NLL map metrics: ", nll_map_metrics)


def change_keys(path_one, path_two):
    pcl_one = pickle.load(open(path_one, 'rb'))
    pcl_two = pickle.load(open(path_two, 'rb'))

    for key in pcl_two['results'].keys():
        if key not in pcl_one['results']:
            if key == '1.0':
                pcl_one['results']['100'] = pcl_two['results'][key]
            else:
                pcl_one['results'][key] = pcl_two['results'][key]

    if '1.0' in pcl_one['results']:
        pcl_one['results']['100'] = pcl_one['results']['1.0']
        del pcl_one['results']['1.0']

    key_name = None
    kk = None
    for key in pcl_one['results'].keys():
        if key[:2] == '0.':
            key_name = key[2:]
            kk = key
            break

    if key_name is not None and kk is not None:
        pcl_one['results'][key_name] = pcl_one['results'][kk]
        del pcl_one['results'][kk]

    with open(path_one, 'wb') as handle:
        pickle.dump(pcl_one, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_paths(path_one, path_two):
    paths_one = [os.path.join(path_one, f'run_{i}') for i in range(5)]
    paths_two = [os.path.join(path_two, f'run_{i}') for i in range(5)]

    paths_one = [os.path.join(p, f'run_number_{i}.pkl') for i, p in enumerate(paths_one)]
    paths_two = [os.path.join(p, f'run_number_{i}.pkl') for i, p in enumerate(paths_two)]

    for p1, p2 in zip(paths_one, paths_two):
        change_keys(p1, p2)


def change_01_to_100(exp_path):
    paths_one = [os.path.join(exp_path, f'run_{i}') for i in range(5)]
    # paths_two = [os.path.join(path_two, f'run_{i}') for i in range(5)]
    paths_one = [os.path.join(p, f'run_number_{i}.pkl') for i, p in enumerate(paths_one)]

    for p in paths_one:
        pcl = pickle.load(open(p, 'rb'))

        if '1.0' in pcl['results']:
            pcl['results']['100'] = pcl['results']['1.0']
            del pcl['results']['1.0']

        with open(p, 'wb') as handle:
            pickle.dump(pcl, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_combined_plot_kfac():

    experiment_paths = [r"C:\Users\45292\Documents\Master\NLP\SST2\laplace",
                        r"C:\Users\45292\Documents\Master\NLP\MRPC\laplace",
                        r"C:\Users\45292\Documents\Master\NLP\RTE\laplace"]

    map_paths = [r"C:\Users\45292\Documents\Master\NLP\SST2\map",
                 r"C:\Users\45292\Documents\Master\NLP\MRPC\map",
                 r"C:\Users\45292\Documents\Master\NLP\RTE\map"]

    val_paths = [r"C:\Users\45292\Documents\Master\NLP\SST2\map_val",
                 r"C:\Users\45292\Documents\Master\NLP\MRPC\map_val",
                 r"C:\Users\45292\Documents\Master\NLP\RTE\map_val"]

    titles = ['SST-2', 'MRPC', 'RTE']
    save_path = r"C:\Users\45292\Documents\Master\Article\Figures"

    fig, ax = plt.subplots(1, 3, figsize = (26,9))

    font = {'family': 'serif',
            'size': 40,
            'serif': 'cmr10'
            }
    mpl.rc('font', **font)
    mpl.rc('legend', fontsize=50)
    mpl.rc('axes', labelsize=60)
    # subset = [True, False, True]
    for i, (map_path, exp_path, val_path) in enumerate(zip(map_paths, experiment_paths, val_paths)):
        ax[i] = make_laplace_plot_three_full(exp_path, map_path=map_path, val_path=val_path, metric='nll',
                                     save_path="", ax = ax[i], subset=i)



    # plt.show()

    for i in range(3):
        ax[i].set_xlabel('Num. Floats', fontsize = 40)
        ax[i].set_title(titles[i], fontsize = 35)
        plt.setp(ax[i].get_xticklabels(), fontsize=30)
        plt.setp(ax[i].get_yticklabels(), fontsize=30)
        ax[i].set_ylabel("")
        ax[i].grid(True, linewidth = 2)
    ax[0].set_ylabel('NLL', fontsize=40)
    plt.subplots_adjust(left=0.08,
                        bottom=0.12,
                        right=0.95,
                        top=0.75,
                        wspace=0.2,
                        hspace=0.1)
    lines = []
    labels = []
    for ax in fig.axes:
        Line, Label = ax.get_legend_handles_labels()
        # print(Label)
        # for line in Line:
            # line.set_linewidth(0)
        lines.extend(Line)
        labels.extend(Label)
        break

    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncols = 3, fontsize = 32)
    fig.savefig(os.path.join(save_path, 'SKFAC-all_datasets.pdf'), format = 'pdf')
    plt.show()



if __name__ == '__main__':


    # make_combined_plot_kfac()
    # breakpoint()
    overall_path = r'C:\Users\45292\Documents\Master\NLP'
    save_path = r'C:\Users\45292\Documents\Master\Article\Figures'
    find_and_dataset_paths_and_plot(overall_path, save_path, type = 'full')
    breakpoint()

    # path = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\sst2\laplace\operator_norm_ramping"
    # save_path = path
    # num_modules = 2
    # read_write_all(path, save_path, num_modules)
    # breakpoint()

    # path = r'C:\Users\45292\Documents\Master\SentimentClassification\Laplace\operator_norm_ramping_prior'
    # path = r'C:\Users\45292\Documents\Master\SentimentClassification\Laplace\random_ramping'
    # path = r'C:\Users\45292\Documents\Master\SentimentClassification\SWAG\random_ramping'

    # map_path = r"C:\Users\45292\Documents\Master\NLP\RTE\map\run_number_0_map.pkl"
    # other_path = r"C:\Users\45292\Documents\Master\NLP\RTE\laplace\nli_sublayer_full\run_0\run_number_0.pkl"
    # data_path =  r"C:\Users\45292\Documents\Master\NLP\RTE\Data\test_data.csv"
    # find_diff_between_map_and_other(map_path, other_path, data_path)
    #
    # breakpoint()
    # # save_path = r'C:\Users\45292\Documents\Master\NLP\SST2\Figures\Examples\Data'
    # # find_diff_between_map_and_other(map_path, other_path, data_path, save_path)
    # # breakpoint()
    # # imdb_map_path = [r"C:\Users\45292\Documents\Master\NLP\RTE\map",
    # #                  r"C:\Users\45292\Documents\Master\NLP\SST2\map",
    # #                  r"C:\Users\45292\Documents\Master\NLP\MRPC\map"]
    # # experiment_path = [r"C:\Users\45292\Documents\Master\NLP\RTE\swag",
    # #                    r"C:\Users\45292\Documents\Master\NLP\SST2\swag",
    # #                    r"C:\Users\45292\Documents\Master\NLP\MRPC\swag"]
    # #
    # # save_path = r'C:\Users\45292\Documents\Master\NLP\Overall Figures'
    # # make_acc_swag_plots_across_datasets(experiment_paths=experiment_path, map_paths=imdb_map_path,
    # #                                     save_path= save_path)
    # # breakpoint()

    experiment_path = r"C:\Users\45292\Documents\Master\NLP\SST2\laplace"
    map_path = r"C:\Users\45292\Documents\Master\NLP\SST2\map"
    save_path = r"C:\Users\45292\Documents\Master\Article\Figures"
    val_path =  r"C:\Users\45292\Documents\Master\NLP\SST2\map_val"
    #
    make_laplace_plot_one(experiment_path, map_path=None, metric='nll', save_path=save_path)
    breakpoint()

    make_laplace_plot_three_full(experiment_path, map_path=map_path, val_path=val_path, metric='nll',
                                 save_path=save_path)
    breakpoint()
    #
    #
    # fig, ax = plt.subplots(1,1)
    # plotter = RampingExperiments(r"C:\Users\45292\Documents\Master\NLP\MRPC\laplace\nli_sublayer_full")
    # map_path = r"C:\Users\45292\Documents\Master\N
    # LP\MRPC\map"
    # val_path =r"C:\Users\45292\Documents\Master\NLP\MRPC\map_val"
    # save_path = r"C:\Users\45292\Documents\Master\NLP\MRPC\Figures\Laplace\calibration_curves.pdf"
    # estimator, map = plotter.collect_predictions_and_calculate_calib(key = '1.0', map_path=map_path)
    # estimator, temp_map = plotter.collect_predictions_and_calculate_calib(key = '1.0', map_path = map_path, val_path=val_path)
    # plotter.plot_calibration(*estimator, ax = ax, color = 'tab:orange', label = 'S-KFAC 0.01 pct')
    # plotter.plot_calibration(*map, ax = ax, color = 'tab:blue', label = 'MAP')
    plotter.plot_calibration(*temp_map, ax = ax, color = 'tab:brown', label = 'Temp. scaled MAP')
    # ax.set_ybound(0, 1.2)
    # ax.legend()
    # ax.set_title('MRPC Calibration Curves')
    # fig.savefig(save_path, format = 'pdf')
    # plt.show()
    # breakpoint()
    #
    # save_path = r'C:\Users\45292\Documents\Master\NLP\RTE\Figures\Swag'
    # exp_paths = [r"C:\Users\45292\Documents\Master\NLP\SST2\laplace\last_layer_full",
    #              r"C:\Users\45292\Documents\Master\NLP\SST2\laplace\last_layer"]
    #
    # experiment_path = r"C:\Users\45292\Documents\Master\NLP\RTE"
    # map_path = r"C:\Users\45292\Documents\Master\NLP\RTE\map"
    # save_path = r"C:\Users\45292\Documents\Master\NLP\RTE\Figures\Laplace"
    #
    # experiment_path = r'C:\Users\45292\Documents\Master\NLP\SST2\laplace'
    # write_ECE_plot_for_dataset(experiment_path, map_path=None, metric='nll')
    # breakpoint()
    # experiment_path = r'C:\Users\45292\Documents\Master\NLP\SST2\laplace'
    # write_ECE_plot_for_dataset(experiment_path, map_path=None, metric='nll')
    # breakpoint()
    # exp_paths = [r"C:\Users\45292\Documents\Master\NLP\SST2\laplace\sublayer_full_fine_grained",
    #              r"C:\Users\45292\Documents\Master\NLP\SST2\laplace\sublayer_full"
    #              ]
    # find_paths(exp_paths[0], exp_paths[1])
    #
    # names = ['Fine grained', 'normal']
    #
    # fig,ax = plt.subplots(1,1)
    # plotter = MultipleRampingExperiments(ramping_exp_paths=exp_paths,
    #                                      ramping_exp_names = names, method = 'Laplace')
    # plotter.plot_all(fig = fig, ax = ax)
    # plt.show()
    # breakpoint()
    # fig, ax = plt.subplots(1,1)
    # # plotter.plot_all(fig, ax)
    # # plt.show()
    # # make_rte_laplace_ll_plot()
    # # make_plot_one_swag(experiment_path, map_path = map_path, save_path=save_path)

    # change_01_to_100(r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\sst2_nll\laplace\sublayer_full")
    # breakpoint()
    experiment_path_laplace = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\laplace"
    map_path = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\map"
    val_path = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\imdb_val_map_nll"
    save_path = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\Figures"
    # make_laplace_plot_one(experiment_path_laplace, map_path=map_path, metric='nll', save_path=save_path)
    # make_laplace_plot_two(experiment_path_laplace, map_path=map_path, save_path=save_path)
    make_laplace_plot_three_full(experiment_path_laplace, map_path=map_path, metric='nll', save_path=save_path, val_path=val_path)
    # make_laplace_plot_three_predefined(experiment_path_laplace, map_path=map_path, save_path=save_path)

    # experiment_path_swag = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\swag"
    # make_plot_one_swag(experiment_path_swag, map_path=map_path, save_path=save_path)
    # make_plot_two_full_swag(experiment_path_swag, map_path=map_path, save_path=save_path)

    # write_ECE_plot_for_dataset(experiment_path_laplace, map_path=None, metric='ECE')
    # write_ECE_plot_for_dataset(experiment_path_swag, map_path=None, metric='ECE')

    breakpoint()


    # breakpoint()
    # make_plot_two_full_swag(experiment_path,map_path=map_path, save_path=save_path)
    # breakpoint()
    # # plotter.write_latex_table(bold_direction='method')
    # # breakpoint()
    # # make_laplace_plot_one(experiment_path, save_path=r'C:\Users\45292\Documents\Master\NLP\RTE\Figures\Laplace')
    # make_plot_two_full_swag(experiment_path, map_path=imdb_map_path, save_path=save_path)
    # # make_plot_two_full_swag(experiment_path, map_path=imdb_map_path, save_path=save_path)
    # breakpoint()
    # make_laplace_plot_three_full(experiment_path, save_path=save_path)
    # make_laplace_plot_one(experiment_path, map_path=imdb_map_path, save_path=save_path)
    # make_laplace_plot_one(experiment_path, save_path=save_path)
    # make_laplace_plot_two(experiment_path, save_path=save_path)
    # breakpoint()
    #
    # root_imdb_laplace_path = r'C:\Users\45292\Documents\Master\SentimentClassification\IMDB\Laplace'
    # exp_paths = [os.path.join(root_imdb_laplace_path, p) for p in os.listdir(root_imdb_laplace_path)
    #              if 'ramping' in p]
    # path = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\laplace\random_ramping_prior"
    # path = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\swag\operator_norm_ramping_subclass"
    # map_path = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\map"

    # exp_paths = [r"C:\Users\45292\Documents\Master\SentimentClassification\Laplace\operator_norm_ramping_subclass_attn_min",
    #             r"C:\Users\45292\Documents\Master\SentimentClassification\Laplace\operator_norm_ramping_subclass_prior"]
    # names = ['Min operator norm attn', 'Max operator norm attn']
    #
    # exp_paths = [r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\swag\sublayer_new",
    #              r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\swag\random_ramping"]

    # root = r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\rte\laplace"
    # exp_paths = [os.path.join(root, p) for p in os.listdir(root)]

    # exp_paths = [r"C:\Users\45292\Documents\Master\SentimentClassification\Laplace\operator_norm_ramping"]

    # map_path = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\imdb\map"

    # map_paths = [r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\mrpc\mrpc_map_acc_100"]
    #
    # names = ["test"]
    # exp_paths = [r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\mrpc\laplace\nli_last_layer"]
    # metrics = ['nll', 'accuracy_score', 'ECE']
    # # name = 'Max operator norm attn'
    # # name_col_path = [("Operator Norm", 'tab:brown', r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\mrpc\swag\nli_operator_norm_ramping_mlp")]
    # # name = 'Op norm'
    #
    # plotter = RampingExperiments(r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\mrpc\laplace\nli_last_layer")
    # plotter.get_map_metrics(map_paths=map_paths, metrics=metrics)

    ### MAP METRICS ###
    write_map_metrics()
    breakpoint()

    # val_path =r"C:\Users\45292\Documents\Master\NLP\MRPC\map_val"
    # save_path = r"C:\Users\45292\Documents\Master\NLP\MRPC\Figures\Laplace\calibration_curves.pdf"
    # estimator, map = plotter.collect_predictions_and_calculate_calib(key = '1.0', map_path=map_path)
    # estimator, temp_map = plotter.collect_predictions_and_calculate_calib(key = '1.0', map_path = map_path, val_path=val_path)
    # for metric in metrics:
    #     plotter = MultipleRampingExperiments(exp_paths, names, map_path=map_path, metric=metric, method='Laplace',
    #                                          sublayer_ramping=False)
    #     fig, ax = plt.subplots(1,1)
    #     plotter.plot_all(fig, ax)
    #     plotter.write_latex_table()

        # plotter.draw_line_at_best(other_path, ax, name = name)
        # plotter.draw_line_at_best(other_path_two, ax, name= name_two, color = 'tab:orange')
        # fig.tight_layout()
        # fig.savefig(os.path.join(save_path, f"{metric}_full.pdf"), format = 'pdf')
        # plt.show()

    breakpoint()
    # fig, ax = plt.subplots(1, 1)
    # plotter = RampingExperiments(path, 'nll')
    # plotter.get_and_plot(path = path, has_seen_softmax = True, ax = ax, map_path=map_path, num_modules=11)
    # plotter.color = 'tab:orange'
    # # plotter.get_and_plot(path = path_, has_seen_softmax=True, ax = ax, map_path=map_path)
    # plt.gcf().subplots_adjust(left=0.16)
    # ylims = ax.get_ylim()
    # diff = (ylims[1] - ylims[0]) * 0.3 + ylims[1]
    #
    # ax.set_ylim((ylims[0], diff))
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
    #           ncol=1, fancybox=True, shadow=True)
    # plt.show()
    # breakpoint()
    #
    #
    #
    # laplace_path = r"C:\Users\45292\Documents\Master\SentimentClassification\Laplace\run_0\run_number_0.pkl"
    # map_path = r"C:\Users\45292\Documents\run_0\run_number_0_map.pkl"
    #
    # lap = pickle.load(open(laplace_path, 'rb'))
    # map = pickle.load(open(map_path, 'rb'))
    # # run_evaluator_again(predictions=map['results'].predictions,labels = map['results'].labels)
    # print(evaluate_loss(map['results'].predictions, map['results'].labels, use_softmax=True))
    # eval_ = Evaluator(map['results'].predictions, map['results'].labels, has_seen_softmax=False)
    # results = {0 : eval_.get_all_metrics()}
    # for key in lap['results'].keys():
    #     eval_ = Evaluator(lap['results'][key].predictions, lap['results'][key].labels, has_seen_softmax=True)
    #     results[key] = eval_.get_all_metrics()
    # breakpoint()
