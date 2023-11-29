import seaborn as sns
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from PartialNLP.Visualization.plot_utils_comb import PlotHelper
from sklearn.linear_model import LinearRegression
import uncertainty_toolbox as uct
from SentimentAnalysis import utils
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
font = {'family': 'serif',
            'size': 15,
            'serif': 'cmr10'
            }
mpl.rc('font', **font)
mpl.rc('legend', fontsize=15)
mpl.rc('axes', labelsize=19)

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
def read_tf_out(path, metrics = ('f1', 'accuracy', 'loss'), use_best = True):

    results = {key: [] for key in metrics}
    metrics = ["/".join(['eval', met]) for met in metrics]

    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag in metrics:
                results[v.tag.split("/")[-1]].append(v.simple_value)

    best_acc = np.argmax(results['accuracy']) if use_best else -1
    results = {key: val[best_acc] for key, val in results.items()}

    return results


def find_file(path):

    paths = os.listdir(path)
    for p in paths:
        if 'event' in p:
            return os.path.join(path, p)
        else:
            return find_file(os.path.join(path, p))

def read_tf_outfiles_for_size_ramping(path, use_best = True):

    paths = os.listdir(path)
    sizes = [float(p.split("_")[-1]) for p in paths]
    results = {}
    for size, p in zip(sizes, paths):
        p = find_file(os.path.join(path, p))
        results[size] = read_tf_out(os.path.join(path, p), use_best=use_best)

    sizes = sorted(sizes)
    accuracies = [results[size]['accuracy'] for size in sizes]
    f1 = [results[size]['f1'] for size in sizes]
    loss = [results[size]['loss'] for size in sizes]

    return accuracies, f1, loss, sizes


def plot_accuracy_f1(accuracy, f1, sizes):

    fig, ax = plt.subplots(1,1)

    sns.pointplot(errorbar=None,
                  x=sizes, y=accuracy,
                  join=False,
                  capsize=.30,
                  markers="d",
                  scale=1.0,
                  err_kws={'linewidth': 0.7},
                  color='tab:blue',
                  label='Accuracy',
                  ax=ax)

    sns.pointplot(errorbar=None,
                  x=sizes, y=f1,
                  join=False,
                  capsize=.30,
                  markers="d",
                  scale=1.0,
                  err_kws={'linewidth': 0.7},
                  color='tab:orange',
                  label='f1 Score',
                  ax=ax)


    ax.set_xlabel('Training Size Percentages')
    ax.set_ylabel('Performance')
    ax.set_title("Performance over data sizes, IMDB")
    fig.tight_layout()
    plt.show()


def get_metrics_from_multiple_runs(path):

    paths = [os.path.join(path, p) for p in os.listdir(path)]
    results = {'accuracy': [], 'f1': [], 'loss': []}
    for p in paths:
        event_path = find_file(p)
        res = read_tf_out(event_path, use_best=False)
        for key, val in res.items():
            results[key].append(val)

    return results


class RampingExperiments:

    def __init__(self, ramping_exp_path, metric = 'nll'):
        self.ramping_exp_path = ramping_exp_path
        self.metric = metric
    def find_files(self, path = None):
        path = self.ramping_exp_path if path is None else path
        files = [os.path.join(path, p) for p in os.listdir(path)]
        run_numbers_and_paths = []
        for file in files:
            if 'run_' in os.path.basename(file) and os.path.isdir(file):
                run_number = int(os.path.basename(file).split("_")[-1])
                run_numbers_and_paths.append((run_number, os.path.join(file, f'run_number_{run_number}.pkl')))

        return sorted(run_numbers_and_paths)
    def get_metrics_from_file(self, file):

        evaluation = utils.read_file(file)
        results = evaluation['results']
        modules = list(results.keys())
        res = {k: [] for k in results[modules[0]].keys()}
        for module in modules:
            for k, v in results[module].items():
                res[k].append(v)
        res['modules'] = modules
        return res

    def get_metrics_from_all_files(self, path = None):

        run_number_and_paths = self.find_files(path)
        results = {}
        for run_number, path in run_number_and_paths:
            results[run_number] = self.get_metrics_from_file(path)

        return results

    def get_specific_results(self, results, key):

        df = pd.DataFrame()
        module_holder, results_holder = [], []

        for run, v in results.items():
            modules = v['modules']
            module_holder+=modules
            results_holder+= v[key]

        df[key] = results_holder
        df['modules'] = module_holder
        return df


    def plot_result(self, df,key, ax = None):
        if ax is None:
            fig, ax = plt.subplots(1,1)

        errorbar_func = lambda x: np.percentile(x, [25, 75])
        sns.pointplot(errorbar=errorbar_func,
                      data=df, x="modules", y=key,
                      join=False,
                      capsize=.30,
                      markers="d",
                      scale=1.0,
                      err_kws={'linewidth': 0.7}, estimator=np.median,
                      color=point_err_color,
                      label=key,
                      ax=ax)

        plt.show()

    def get_and_plot(self):
        results = self.get_metrics_from_all_files(self.ramping_exp_path)
        key = self.metric
        df = self.get_specific_results(results, key)
        self.plot_result(df, key)





if __name__ == '__main__':


    path = r'C:\Users\45292\Documents\Master\SentimentClassification\Laplace\operator_norm_ramping'

    plotter = RampingExperiments(path)
    plotter.get_and_plot()
    breakpoint()

    path = r'C:\Users\45292\Documents\Master\SentimentClassification\datasize_ramping\imdb_train_size'

    accuracies, f1, loss, sizes = read_tf_outfiles_for_size_ramping(path)
    # p2 = r'C:\Users\45292\Documents\Master\SentimentClassification\runs_100\IMDB'
    # final_results = get_metrics_from_multiple_runs(p2)

    accuracies = accuracies[:-1]
    f1 = f1[:-1]
    accuracies.append(0.9281)
    f1.append(0.9283)
    plot_accuracy_f1(accuracies, f1, sizes)
    breakpoint()