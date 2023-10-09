import seaborn as sns
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt


def plot_errorbar_percentages(df, errorbar_func, estimator=None, ax=None, y='la_nll'):
    sns.pointplot(errorbar=errorbar_func,
                  data=df, x="percentages", y=y,
                  join=False,
                  markers="d", scale=.5, err_kws={'linewidth': 0.5}, estimator=estimator,
                  ax=ax)

    nll_array = np.array(df[y]).reshape((-1, 10))
    estimated = estimator(nll_array, axis=0)
    map_val = estimated[0]
    ax.axhline(y=map_val, linestyle='--', linewidth=1, alpha=0.7, label=None)
    fully_stochastic_val = estimated[-1]
    ax.axhline(y=fully_stochastic_val, linestyle='--', linewidth=1, alpha=0.7, label=None)

    # Adjust layout
    plt.tight_layout()


def plot_estimator(df, errorbar_func, estimator=None, ax=None, data_name=None):
    legend_labels = []
    for key in df:
        if 'percentages' not in key:
            legend_labels.append(key.split('_')[0])
            plot_errorbar_percentages(df=df,
                                      errorbar_func=errorbar_func,
                                      estimator=estimator,
                                      y=key,
                                      ax=ax)

    ax.set_title("Laplace Swag" + " - " + estimator.__name__ + " - " + data_name, fontsize=12, pad=-20)

    # Set labels and legend
    ax.set_xlabel("Percentages", fontsize=12)
    ax.set_ylabel("nll", fontsize=12)
    # ax.legend(title="Methods", labels=legend_labels, fontsize=10)

    # Customize the grid appearance
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    # ax.grid(b=True, which='major', color='w', linewidth=1.0)

    plt.show(block=False)


def plot_partial_percentages(percentages, res, data_name=None, df=None, num_runs=15):
    if df is None:
        df = pd.DataFrame()
        df['percentages'] = num_runs * percentages
        for key, val in res.items():
            if 'la' in key:
                df[key + '_nll'] = [nll.item() for nll in val.flatten()]
            elif 'swag' in key:
                df[key + '_nll'] = [(-1) * ll for ll in val.flatten()]
    fig, ax = plt.subplots(1, 1)
    # plt.rcParams['figure.figsize'] = [10, 8]

    # Plot median
    # plot_estimator(df=df, errorbar_func=lambda x: np.percentile(x, (25, 75)), estimator=np.median, ax=ax, data_name=data_name)

    # Plot mean
    plot_estimator(df=df, errorbar_func=('ci', 100), estimator=np.mean, ax=ax, data_name=data_name)
    breakpoint()

def plot_series(percentages, res, title = None):
    fig, ax = plt.subplots(1,1)
    perc = percentages
    rs = res - res.mean(-1)[:, None]

    runs = np.zeros_like(rs)
    for i in range(runs.shape[0]):
        runs[i] = i + 1
    df = pd.DataFrame()
    df['runs'] = runs.flatten()
    df['nll'] = rs.flatten()

    df['percentages'] = runs.shape[0] * perc
    sns.lineplot(data=df, x='percentages', y='nll', errorbar=None, ax = ax, linewidth = 2, legend=False)
    sns.lineplot(data=df, x = 'percentages', y = 'nll',
                 hue ='runs', style = 'runs', alpha = 0.4, ax = ax, legend=False,
                 palette = sns.color_palette(['black']))
    for line in ax.lines[1:]:
        line.set(linestyle = '-.')
    ax.lines[0].set_label('Mean')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    plt.show()


def read_data_swag_la(path, include_map = True):
    percentages = []
    test_nll = []
    test_mse = []
    for p in os.listdir(path):
        if 'results' in p:
            pcl = pickle.load(open(os.path.join(path, p), 'rb'))
            if not include_map:
                test_nll.append(pcl['test_nll'][1:])
            else:
                test_nll.append(pcl['test_nll'])
            percentages.append(pcl['percentages'])
            test_mse.append([i.item() if not isinstance(i, float) else i for i in pcl['test_mse']][1:])


    percentages = percentages[-1]
    if include_map:
        percentages = [0] + percentages
    test_nll = np.array(test_nll)
    test_mse = np.array(test_mse)
    return percentages, test_nll, test_mse


def read_vi_data(path):

    percentages = []
    test_nll = []
    test_mse = []
    for p in os.listdir(path):
        run_number = int(p.split("_")[-2])
        pcl = pickle.load(open(os.path.join(path, p), 'rb'))
        test_nll.append(pcl['test_nll'])
        percentages.append(pcl['percentages'])
        test_mse.append([i.item() if not isinstance(i, float) else i for i in pcl['val_mse']])

    percentages = percentages[-1]
    test_nll = np.array(test_nll)
    test_mse = np.array(test_mse)
    return percentages, test_nll, test_mse


def read_hmc_data(path):
    percentages = ['map_results', '1', '2', '5', '8', '14', '23', '37', '61', '100']
    test_nll = {key: [] for key in percentages}
    test_mse = {key: [] for key in percentages}

    for p in os.listdir(path):
        pcl = pickle.load(open(os.path.join(path, p), 'rb'))

        for perc in percentages:
            if perc in pcl:
                test_nll[perc].append(pcl[perc]['test_ll'])
                test_mse[perc].append(pcl[perc]['test_rmse'])

    dfnll = dict_to_df(test_nll)
    dfmse = dict_to_df(test_mse)

    return dfnll, dfmse


def dict_to_df(dic):
    dfnll = pd.DataFrame()
    percentages, data = [], []
    for key, val in dic.items():
        percentages += [0]*len(val) if 'map' in key else [int(key)] * len(val)
        data += val
    dfnll['percentages'] = percentages
    dfnll['nll'] = [- i for i in data]
    return dfnll


def get_under_folders_and_names(path):
    names = []
    paths = []
    for p in os.listdir(path):
        if 'model' in p:
            names.append(p.split("_")[0])
            paths.append(os.path.join(path, p))
    return names, paths

def plot_hmc_vi(path_hmc, path_vi):
    names, paths_hmc = get_under_folders_and_names(path_hmc)
    _, paths_vi = get_under_folders_and_names(path_vi)

    for name, p_la, p_swag in zip(names, path_hmc, path_vi):
        # dfnll, dfmse = read_hmc_data(path)
        percentages, test_nll_la, test_mse_la = read_data_swag_la(p_la, include_map=True)
        _, test_nll_swag, test_mse_swag = read_data_swag_la(p_swag, include_map=True)
        plot_partial_percentages(percentages=percentages,
                                 res={'la': test_nll_la, 'swag': test_nll_swag},
                                 data_name=name,
                                 num_runs=test_nll_la.shape[0])


def plot_la_swag(path_la, path_swag):
    names, paths_la = get_under_folders_and_names(path_la)
    _, paths_swag = get_under_folders_and_names(path_swag)

    for name, p_la, p_swag in zip(names, paths_la, paths_swag):
        # dfnll, dfmse = read_hmc_data(path)
        percentages, test_nll_la, test_mse_la = read_data_swag_la(p_la, include_map=True)
        _, test_nll_swag, test_mse_swag = read_data_swag_la(p_swag, include_map=True)
        plot_partial_percentages(percentages=percentages,
                                 res={'la': test_nll_la, 'swag': test_nll_swag},
                                 data_name=name,
                                 num_runs=test_nll_la.shape[0])


if __name__ == '__main__':
    path_la = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_Laplace_MAP'
    path_swag = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_SWAG_MAP_nobayes'
    plot_la_swag(path_la, path_swag)
    breakpoint()


