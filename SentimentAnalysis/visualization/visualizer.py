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
from tensorflow.python.summary.summary_iterator import summary_iterator
font = {'family': 'serif',
            'size': 15,
            'serif': 'cmr10'
            }
mpl.rc('font', **font)
mpl.rc('legend', fontsize=15)
mpl.rc('axes', labelsize=19)


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

if __name__ == '__main__':

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