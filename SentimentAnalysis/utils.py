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


def save_laplace(save_path, laplace_cfg, **kwargs):
    pass


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
              df_combined.to_latex(index=True, float_format="{:.2f}".format),
              '\label{} \n', r"\end{table}", "\n")

    def draw_line_at_best(self, other_path, ax, name=None, color=None, num_modules=None, best_mod=True,
                          set_point_instead=False, sublayer_ramping=False):

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

        if name != 'Last layer':
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
            scores['modules'] = [100 for _ in range(len(scores))]
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

            ticks = ax.get_xmajorticklabels()
            ticks[-1].set_text('100')
            ax.set_xticklabels(ticks)
        else:

            ax.axhline(y=best_score, linestyle='--', linewidth=1.5, alpha=0.7,
                       color='tab:red' if color is None else color, label=label_name)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                      ncol=1, fancybox=True, shadow=True)
        return ax

    def get_dataset_name(self, path):
        if 'imdb' in path.lower():
            return 'IMDB'
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

    def plot_all(self, fig=None, ax=None):
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
            name = self.path_to_names[key]
            color = self.exp_number_to_colors[idx]

            sns.pointplot(errorbar=error_bar_percentile,
                          data=df, x="modules", y=self.metric,
                          join=False,
                          capsize=.30,
                          markers="d",
                          scale=1.0,
                          err_kws={'linewidth': 0.7}, estimator=np.median,
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

        y_label = self.metric_to_label_metric[self.metric]
        for number, tick in zip(numbers, ticks):
            tick.set_text(number)

        ax.set_xticklabels(ticks)
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
            results_holder += v[key]

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


def make_laplace_plot_one(experiment_path, map_path=None, save_path="", metric = 'nll'):
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)
    names = ['Operator norm MLP', 'Operator norm attn', 'Random ramping']
    keys = ['operator_norm_ramping_mlp', 'operator_norm_ramping_attn', 'random_ramping']
    experiment_paths = [experiment_to_paths[key] for key in keys]

    last_layer_path = experiment_to_paths['last_layer_full']
    last_layer_name = 'Last layer full'

    plotter = MultipleRampingExperiments(experiment_paths, names, map_path,metric = metric, method='Laplace')
    fig, ax = plt.subplots(1, 1)
    plotter.plot_all(fig=fig, ax=ax)
    plotter.draw_line_at_best(last_layer_path, ax, last_layer_name, color='tab:green', best_mod=False)
    last_layer_path = experiment_to_paths['last_layer']
    last_layer_name = 'Last layer KFAC'
    plotter.draw_line_at_best(last_layer_path, ax, last_layer_name, color='tab:brown', best_mod=False)
    fig.tight_layout()
    if save_path:
        if map_path is not None:
            save_name = f"laplace_plot_three_full_{metric}_w_map.pdf"
            fig.savefig(os.path.join(save_path, save_name))
        else:
            save_name = f"laplace_plot_three_full_{metric}.pdf"
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
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)
    if 'RTE' in path:
        names = ['Random ramping', 'Last layer']
        keys = ['random_ramping', 'last_layer']

    elif 'SST2' in path:
        names = ['Random ramping', 'Last layer']
        keys = ['random_ramping', 'last_layer']

    elif 'MRPC' in path:
        names = ['Random ramping', 'Last layer']
        keys = ['random_ramping', 'last_layer']
    else:
        raise ValueError("Could not decipher path")
    return names, keys


def choose_right_ones_speficic(path):
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)

    if 'RTE' in path:
        names = ['Max operator norm attn. + LL', 'S-KFAC full model', 'Last layer']
        keys = ['operator_norm_ramping_attn_ll', 'sublayer_full', 'last_layer']

    elif 'SST2' in path:
        names = ['Max operator norm MLP', 'S-KFAC full model', 'Last layer']
        keys = ['operator_norm_ramping_mlp', 'sublayer_full', 'last_layer']

    elif 'MRPC' in path:
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

def make_laplace_plot_three_full(experiment_path, map_path=None, save_path="", metric = 'nll', val_path = None):
    experiment_to_paths = make_experiment_to_path_mapping(experiment_path)

    names = ['S-KFAC full model']
    keys = ['sublayer_full']

    exp_paths = [experiment_to_paths[key] for key in keys]

    colors = ['tab:green', 'tab:orange', 'tab:brown']
    names_, keys_ = choose_right_ones_full(experiment_path)

    plotter = MultipleRampingExperiments(exp_paths, names, map_path if val_path is None else None,
                                         sublayer_ramping=True, metric = metric, method='Laplace')

    fig, ax = plt.subplots(1, 1)
    plotter.plot_all(fig=fig, ax=ax)
    for idx, (n, k) in enumerate(zip(names_, keys_)):
        p = experiment_to_paths[k]
        col = colors[idx]
        plotter.draw_line_at_best(p, ax, name=n, color=col)

    if val_path is not None and map_path is not None:
        nlls, eces = include_temperature_scaling(plotter, map_path, val_path)
        best_score = np.median(nlls) if metric == 'nll' else np.median(eces)
        ybounds = ax.get_ybound()
        diff = ybounds[1] - ybounds[0]
        if ybounds[1] < (best_score - diff / 4):
            y1 = best_score + diff / 3
            ax.set_ybound(ybounds[0], y1)

        color = 'tab:red'
        label_name = 'Temp. scaled MAP'
        ax.axhline(y=best_score, linestyle='--', linewidth=1.5, alpha=0.7,
                   color='tab:red' if color is None else color, label=label_name)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                  ncol=1, fancybox=True, shadow=True)


    fig.tight_layout()
    if save_path:
        if map_path is not None and val_path is None:
            save_name = f"laplace_plot_three_full_{metric}_w_map.pdf"
            fig.savefig(os.path.join(save_path, save_name))
        else:
            save_name = f"laplace_plot_three_full_{metric}.pdf"
            fig.savefig(os.path.join(save_path, save_name))

    plt.show()


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


def write_ECE_plot_for_dataset(experiment_path, map_path):

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

    plotter_laplace = MultipleRampingExperiments(laplace_paths, map_path=map_path,metric='ECE')
    plotter_swag = MultipleRampingExperiments(swag_paths, map_path=map_path, metric='ECE')
    plotter_laplace.write_latex_table(bold_direction='method')
    plotter_swag.write_latex_table(bold_direction='method')



if __name__ == '__main__':

    # path = r"C:\Users\Gustav\Desktop\MasterThesisResults\SentimentAnalysis\sst2\laplace\operator_norm_ramping"
    # save_path = path
    # num_modules = 2
    # read_write_all(path, save_path, num_modules)
    # breakpoint()

    # path = r'C:\Users\45292\Documents\Master\SentimentClassification\Laplace\operator_norm_ramping_prior'
    # path = r'C:\Users\45292\Documents\Master\SentimentClassification\Laplace\random_ramping'
    # path = r'C:\Users\45292\Documents\Master\SentimentClassification\SWAG\random_ramping'

    map_path = r"C:\Users\45292\Documents\Master\NLP\MRPC\map\run_number_0_map.pkl"
    other_path = r"C:\Users\45292\Documents\Master\NLP\MRPC\swag\nli_random_ramping\run_0\run_number_0.pkl"
    data_path =  r"C:\Users\45292\Documents\Master\NLP\MRPC\Data\test_data.csv"
    find_diff_between_map_and_other_swag(map_path, other_path, data_path)

    breakpoint()
    # save_path = r'C:\Users\45292\Documents\Master\NLP\SST2\Figures\Examples\Data'
    # find_diff_between_map_and_other(map_path, other_path, data_path, save_path)
    # breakpoint()
    # imdb_map_path = [r"C:\Users\45292\Documents\Master\NLP\RTE\map",
    #                  r"C:\Users\45292\Documents\Master\NLP\SST2\map",
    #                  r"C:\Users\45292\Documents\Master\NLP\MRPC\map"]
    # experiment_path = [r"C:\Users\45292\Documents\Master\NLP\RTE\swag",
    #                    r"C:\Users\45292\Documents\Master\NLP\SST2\swag",
    #                    r"C:\Users\45292\Documents\Master\NLP\MRPC\swag"]
    #
    # save_path = r'C:\Users\45292\Documents\Master\NLP\Overall Figures'
    # make_acc_swag_plots_across_datasets(experiment_paths=experiment_path, map_paths=imdb_map_path,
    #                                     save_path= save_path)
    # breakpoint()
    # experiment_path = r"C:\Users\45292\Documents\Master\NLP\RTE\laplace"
    # map_path = r"C:\Users\45292\Documents\Master\NLP\RTE\map"
    # save_path = r"C:\Users\45292\Documents\Master\NLP\RTE\Figures\Laplace"
    # val_path =  r"C:\Users\45292\Documents\Master\NLP\RTE\map_val"
    # make_laplace_plot_three_full(experiment_path, map_path, val_path=val_path, metric='ECE', save_path = save_path)
    # breakpoint()


    fig, ax = plt.subplots(1,1)
    plotter = RampingExperiments(r"C:\Users\45292\Documents\Master\NLP\MRPC\laplace\nli_sublayer_full")
    map_path = r"C:\Users\45292\Documents\Master\NLP\MRPC\map"
    val_path =r"C:\Users\45292\Documents\Master\NLP\MRPC\map_val"
    save_path = r"C:\Users\45292\Documents\Master\NLP\MRPC\Figures\Laplace\calibration_curves.pdf"
    estimator, map = plotter.collect_predictions_and_calculate_calib(key = '1.0', map_path=map_path)
    estimator, temp_map = plotter.collect_predictions_and_calculate_calib(key = '1.0', map_path = map_path, val_path=val_path)
    plotter.plot_calibration(*estimator, ax = ax, color = 'tab:orange', label = 'S-KFAC 0.01 pct')
    plotter.plot_calibration(*map, ax = ax, color = 'tab:blue', label = 'MAP')
    plotter.plot_calibration(*temp_map, ax = ax, color = 'tab:brown', label = 'Temp. scaled MAP')
    ax.set_ybound(0, 1.2)
    ax.legend()
    ax.set_title('MRPC Calibration Curves')
    fig.savefig(save_path, format = 'pdf')
    plt.show()
    breakpoint()

    save_path = r'C:\Users\45292\Documents\Master\NLP\RTE\Figures\Swag'
    exp_paths = [r"C:\Users\45292\Documents\Master\NLP\SST2\laplace\last_layer_full",
                 r"C:\Users\45292\Documents\Master\NLP\SST2\laplace\last_layer"]

    experiment_path = r"C:\Users\45292\Documents\Master\NLP\RTE"
    map_path = r"C:\Users\45292\Documents\Master\NLP\RTE\map"
    save_path = r"C:\Users\45292\Documents\Master\NLP\RTE\Figures\Laplace"

    write_ECE_plot_for_dataset(experiment_path, map_path=None)
    breakpoint()
    # plotter = MultipleRampingExperiments(ramping_exp_paths=exp_paths,
    #                                      ramping_exp_names = ['last layer full', 'normal'], method = 'Laplace')
    # fig, ax = plt.subplots(1,1)
    # plotter.plot_all(fig, ax)
    # plt.show()
    # make_rte_laplace_ll_plot()
    # make_plot_one_swag(experiment_path, map_path = map_path, save_path=save_path)
    make_laplace_plot_three_full(experiment_path,map_path=map_path, metric='ECE', save_path=save_path)
    # make_laplace_plot_one(experiment_path,map_path=map_path, metric = 'ECE')

    breakpoint()
    make_plot_two_full_swag(experiment_path,map_path=map_path, save_path=save_path)
    breakpoint()
    # plotter.write_latex_table(bold_direction='method')
    # breakpoint()
    # make_laplace_plot_one(experiment_path, save_path=r'C:\Users\45292\Documents\Master\NLP\RTE\Figures\Laplace')
    make_plot_two_full_swag(experiment_path, map_path=imdb_map_path, save_path=save_path)
    # make_plot_two_full_swag(experiment_path, map_path=imdb_map_path, save_path=save_path)
    breakpoint()
    make_laplace_plot_three_full(experiment_path, save_path=save_path)
    make_laplace_plot_one(experiment_path, map_path=imdb_map_path, save_path=save_path)
    make_laplace_plot_one(experiment_path, save_path=save_path)
    make_laplace_plot_two(experiment_path, save_path=save_path)
    breakpoint()

    root_imdb_laplace_path = r'C:\Users\45292\Documents\Master\SentimentClassification\IMDB\Laplace'
    exp_paths = [os.path.join(root_imdb_laplace_path, p) for p in os.listdir(root_imdb_laplace_path)
                 if 'ramping' in p]
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

    metrics = ['accuracy_score']
    # name = 'Max operator norm attn'
    # name_col_path = [("Operator Norm", 'tab:brown', r"C:\Users\Gustav\Desktop\MasterThesisResults\NLI\mrpc\swag\nli_operator_norm_ramping_mlp")]
    # name = 'Op norm'

    for metric in metrics:
        plotter = MultipleRampingExperiments(exp_paths, names, map_path=map_path, metric=metric, method='Laplace',
                                             sublayer_ramping=False)
        # fig, ax = plt.subplots(1,1)
        # plotter.plot_all(fig, ax)
        plotter.write_latex_table()

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
