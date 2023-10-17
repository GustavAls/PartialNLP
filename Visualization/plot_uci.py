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

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_errorbar_percentages(df, errorbar_func, estimator=None, ax=None, y='Laplace_nll', color_scheme_1=True):
    # cmap = get_cmap(20)
    # color_num = np.random.randint(2, 18)
    # color_map = mpl.colormaps['Set1']

    map_color = 'tab:red'
    stochastic_color = 'tab:green'
    point_err_color = 'tab:blue' if color_scheme_1 else 'tab:orange'

    sns.pointplot(errorbar=errorbar_func,
                  data=df, x="percentages", y=y,
                  join=False,
                  capsize=.30 if color_scheme_1 else 0.15,
                  markers="d",
                  scale=1.0 if color_scheme_1 else 0.7,
                  err_kws={'linewidth': 0.7}, estimator=estimator,
                  color=point_err_color,
                  label=y.split('_')[0],
                  ax=ax)

    nll_array = np.array(df[y]).reshape((-1, 10))
    estimated = estimator(nll_array, axis=0)
    map_val = estimated[0]
    ax.axhline(y=map_val, linestyle='--', linewidth=1, alpha=0.7,
               color=map_color, label='MAP' if not color_scheme_1 else '_nolegend_')
    fully_stochastic_val = estimated[-1]
    ax.axhline(y=fully_stochastic_val, linestyle='--', linewidth=1, alpha=0.7,
               color=stochastic_color, label='100% Stochastic' if not color_scheme_1 else '_nolegend_')

    # Adjust layout
    plt.tight_layout()


def plot_estimator(df, errorbar_func, estimator=None, ax=None, data_name=None):
    method_names = []
    np.random.seed(42)
    color_scheme_1 = True
    for i, key in enumerate(df):
        if 'percentages' not in key:
            method_names.append(key.split('_')[0])
            plot_errorbar_percentages(df=df,
                                      errorbar_func=errorbar_func,
                                      estimator=estimator,
                                      y=key,
                                      ax=ax,
                                      color_scheme_1=color_scheme_1)
            color_scheme_1 = not color_scheme_1

    title = method_names[0] + " & " + method_names[1] + " - " + estimator.__name__ + " - " + data_name
    ax.set_title(label=title, fontsize=12, pad=-20)

    # Set labels and legend
    ax.set_xlabel("Percentages", fontsize=12)
    ax.set_ylabel("nll", fontsize=12)
    ax.legend(fontsize=10)

    # Customize the grid appearance
    # ax.grid(axis='y', linestyle='-', alpha=0.3)
    # ax.grid(b=True, which='major', color='w', linewidth=1.0)
    # Set the Seaborn style
    plt.savefig(os.path.join(os.getcwd(), title + '.png'))
    plt.show(block=False)


def plot_partial_percentages(percentages, res, data_name=None, df=None, num_runs=15):
    if df is None:
        df = pd.DataFrame()
        df['percentages'] = num_runs * percentages
        for key, val in res.items():
            if 'Laplace' in key:
                df[key + '_nll'] = [nll.item() for nll in val.flatten()]
            else:
                df[key + '_nll'] = [(-1) * ll for ll in val.flatten()]


    # Plot median
    fig, ax = plt.subplots(1, 1)
    plot_estimator(df=df, errorbar_func=lambda x: np.percentile(x, (25, 75)), estimator=np.median, ax=ax, data_name=data_name)

    # Plot mean
    fig, ax = plt.subplots(1, 1)
    plot_estimator(df=df, errorbar_func=('ci', 100), estimator=np.mean, ax=ax, data_name=data_name)

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


def read_data_swag_la_combined(path, include_map=True):
    # TODO: NLL name correction, needs correction in uci_laplace.py for correct keys
    percentages = []
    test_nll_la = []
    test_mse_la = []
    test_nll_swag = []
    test_mse_swag = []
    for p in os.listdir(path):
        if 'results' in p:
            pcl = pickle.load(open(os.path.join(path, p), 'rb'))
            if 'laplace' in p:
                test_nll_la.append(pcl['test_nll'])
                percentages.append(pcl['percentages'])
                test_mse_la.append([i.item() if not isinstance(i, float) else i for i in pcl['test_mse']][1:])
            else:
                test_nll_swag.append(pcl['test_nll'])
                test_mse_swag.append([i.item() if not isinstance(i, float) else i for i in pcl['test_mse']][1:])

    percentages = percentages[-1]
    if include_map:
        percentages = [0] + percentages
    test_nll_la = np.array(test_nll_la)
    test_mse_la = np.array(test_mse_la)
    test_nll_swag = np.array(test_nll_swag)
    test_mse_swag = np.array(test_mse_swag)
    return percentages, test_nll_la, test_mse_la, test_nll_swag, test_mse_swag


def read_vi_data(path):
    percentages = []
    test_ll = []
    test_mse = []
    for p in os.listdir(path):
        pcl = pickle.load(open(os.path.join(path, p), 'rb'))
        test_ll.append(pcl['test_ll'])
        percentages.append(pcl['percentiles'])
    percentages = percentages[-1]
    test_ll = np.array(test_ll)
    return percentages, test_ll, test_mse


def read_hmc_data(path):
    percentages = ['map_results', '1', '2', '5', '8', '14', '23', '37', '61', '100']
    test_ll = {key: [] for key in percentages}
    test_mse = {key: [] for key in percentages}

    for p in os.listdir(path):
        pcl = pickle.load(open(os.path.join(path, p), 'rb'))
        for perc in percentages:
            if perc in pcl:
                test_ll[perc].append(pcl[perc]['test_ll'])
                test_mse[perc].append(pcl[perc]['test_rmse'])

    test_ll = dict_to_nparray(test_ll)
    test_mse = dict_to_nparray(test_mse)

    return test_ll, test_mse


def read_hmc_vi_combined(path):
    percentages = []
    test_ll_vi = []
    hmc_percentiles = ['map_results', '1', '2', '5', '8', '14', '23', '37', '61', '100']
    test_ll_hmc = {key: [] for key in hmc_percentiles}

    for p in os.listdir(path):
        pcl = pickle.load(open(os.path.join(path, p), 'rb'))
        if 'hmc' in p:
            for perc in hmc_percentiles:
                if perc in pcl:
                    test_ll_hmc[perc].append(pcl[perc]['test_ll'])
        else:
            test_ll_vi.append(pcl['test_ll'])
            percentages.append(pcl['percentiles'])

    percentages = percentages[-1]
    test_ll_vi = np.array(test_ll_vi)
    test_ll_hmc = dict_to_nparray(test_ll_hmc)

    return percentages, test_ll_vi, test_ll_hmc


def dict_to_nparray(dic):
    percentages, data = [], []
    for key, val in dic.items():
        percentages.append([0]*len(val) if 'map' in key else [int(key)] * len(val))
        data.append(val)
    data = np.transpose(np.array(data))
    return data


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

    for name, p_hmc, p_vi in zip(names, paths_hmc, paths_vi):
        test_ll_hmc, test_mse_hmc = read_hmc_data(p_hmc)
        percentages, test_ll_vi, test_mse_vi = read_vi_data(p_vi)
        plot_partial_percentages(percentages=percentages,
                                 res={'HMC': test_ll_hmc, 'VI': test_ll_vi},
                                 data_name=name,
                                 num_runs=test_ll_vi.shape[0])


def plot_hmc_vi_combined(path):
    names, paths = get_under_folders_and_names(path)
    for name, p in zip(names, paths):
        percentages, test_ll_vi, test_ll_hmc = read_hmc_vi_combined(p)
        plot_partial_percentages(percentages=percentages,
                                 res={'HMC': test_ll_hmc, 'VI': test_ll_vi},
                                 data_name=name,
                                 num_runs=test_ll_vi.shape[0])


def plot_la_swag(path_la, path_swag):
    names, paths_la = get_under_folders_and_names(path_la)
    _, paths_swag = get_under_folders_and_names(path_swag)

    for name, p_la, p_swag in zip(names, paths_la, paths_swag):
        percentages, test_nll_la, test_mse_la = read_data_swag_la(p_la, include_map=True)
        _, test_ll_swag, test_mse_swag = read_data_swag_la(p_swag, include_map=True)
        plot_partial_percentages(percentages=percentages,
                                 res={'Laplace': test_nll_la, 'SWAG': test_ll_swag},
                                 data_name=name,
                                 num_runs=test_nll_la.shape[0])


def plot_la_swag_combined(path):
    names, paths = get_under_folders_and_names(path)

    for name, p in zip(names, paths):
        (percentages,
         test_nll_la,
         test_mse_la,
         test_ll_swag,
         test_mse_swag) = read_data_swag_la_combined(p, include_map=True)
        plot_partial_percentages(percentages=percentages,
                                 res={'Laplace': test_nll_la, 'SWAG': test_ll_swag},
                                 data_name=name,
                                 num_runs=test_nll_la.shape[0])




if __name__ == '__main__':
    # path_la = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_Laplace_MAP'
    # path_swag = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_SWAG_MAP_nobayes'
    # plot_la_swag(path_la, path_swag)

    # path = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_Laplace_SWAG_1'
    # plot_la_swag_combined(path)

    # plot_la_swag(path_la, path_swag)


    # HMC VI
    # path_hmc = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_HMC'
    # path_vi = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_VI'
    # plot_hmc_vi(path_hmc, path_vi)

    path_hmc_vi = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_HMC_VI'
    plot_hmc_vi_combined(path_hmc_vi)

    breakpoint()


