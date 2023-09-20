import seaborn as sns
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt
def plot_stuff(percentages, res):
    percentages = [0] + percentages
    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame()
    df['percentages'] = res.shape[0]*percentages
    df['nll'] = res.flatten() * (-1)
    plt.figure()
    sns.pointplot(errorbar=lambda x: np.percentile(x, [25, 75]),
                  data=df, x="percentages", y="nll",
                  join=False,
                  markers="d", scale=.5, errwidth=0.5, estimator=np.median)
    plt.show(block = False)
    plt.figure()
    sns.pointplot(data=df, x="percentages", y="nll",
                  join=False, errorbar=('ci', 50),
                  markers="d", scale=.5, errwidth=0.5)
    plt.show(block=False)


def plot_series(percentages, res):
    fig, ax = plt.subplots(1,1)
    perc = [0] + percentages
    rs = res - res.mean(-1)[:, None]
    rs *= -1

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
    plt.show()

def read_data_swag(path):

    percentages = []
    test_nll = []
    test_mse = []
    for p in os.listdir(path):
        run_number = int(p.split("_")[-1].split(".")[0])
        if run_number != 8:
            pcl = pickle.load(open(os.path.join(path, p), 'rb'))
            test_nll.append(pcl['test_nll'])
            percentages.append(pcl['percentages'])
            test_mse.append([i.item() if not isinstance(i, float) else i for i in pcl['test_mse']])

    percentages = percentages[-1]
    test_nll = np.array(test_nll)
    test_mse = np.array(test_mse)
    return percentages, test_nll, test_mse

if __name__ == '__main__':
    path = r'C:\Users\45292\Documents\Master\Swag Simple\UCI\Results Full\UCI_SWAG\yacht_models'
    percentages, test_nll, test_mse = read_data_swag(path)
    plot_series(percentages, test_nll)
    plot_stuff(percentages, test_nll)

    plot_series(percentages, test_mse)
    plot_stuff(percentages, test_mse)
    breakpoint()