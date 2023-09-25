import seaborn as sns
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt
def plot_stuff(percentages, res, title = None, df = None):
    percentages = percentages
    import seaborn as sns
    import pandas as pd
    if df is None:
        df = pd.DataFrame()
        df['percentages'] = res.shape[0]*percentages
        df['nll'] = res.flatten()
    fig, ax = plt.subplots(1,1)
    sns.pointplot(errorbar=lambda x: np.percentile(x, [25, 75]),
                  data=df, x="percentages", y="nll",
                  join=False,
                  markers="d", scale=.5, errwidth=0.5, estimator=np.median,
                  ax = ax)
    if title is not None:
        ax.set_title('Median Estimator ' + title)
    plt.show(block = False)
    fig, ax = plt.subplots(1,1)
    sns.pointplot(data=df, x="percentages", y="nll",
                  join=False, errorbar=('ci', 50),
                  markers="d", scale=.5, errwidth=0.5,
                  ax = ax)
    if title is not None:
        ax.set_title('Mean Estimator' + title)
    plt.show(block=False)


def plot_series(percentages, res, title = None):
    fig, ax = plt.subplots(1,1)
    perc = percentages
    rs = res - res.mean(-1)[:, None]

    # rs *= -1

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

def read_data_swag(path):

    percentages = []
    test_nll = []
    test_mse = []
    for p in os.listdir(path):
        if 'results' in p:

            pcl = pickle.load(open(os.path.join(path, p), 'rb'))
            test_nll.append(pcl['test_nll'][1:])
            percentages.append(pcl['percentages'])
            test_mse.append([i.item() if not isinstance(i, float) else i for i in pcl['test_mse']][1:])

    percentages = percentages[-1]
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


    test_mse = []

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
if __name__ == '__main__':


    path = r'C:\Users\45292\Documents\Master\HMC\UCI_HMC\yacht_models'
    dfnll, dfmse = read_hmc_data(path)

    # plot_series(percentages, -test_nll[:, 1:], title='Boston Swag')
    plot_stuff(percentages = 0, res = None, title='Yacht', df = dfnll)
    #
    # plot_series(percentages, test_mse)
    # plot_stuff(percentages, test_mse)
    breakpoint()