import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import pickle
plt.rcParams["font.family"] = "Times New Roman"

custom_colors = ['tab:gray']
custom_colors.extend([sns.color_palette('colorblind')[1]]*8)
custom_colors.append(sns.color_palette('colorblind')[0])

def format_plot(ylabel=False, ylim=None, num_params = 7, labels = None):
    plt.xlabel("Percentage of parameters\nsampled (%)", fontsize=6)
    if ylabel:
        plt.ylabel("Average negative test\n log likelihood $(\\downarrow)$", fontsize=7)
    else:
        y=plt.ylabel("")
    plt.xticks(np.arange(num_params), labels, fontsize=5)
    # plt.yscale('symlog')
    if ylim:
        plt.ylim(ylim)
    plt.yticks(fontsize=6)


def create_plot_from_df(subplot_df):
    def iqr(x):
        return np.percentile(x, [25, 75])

    sns.pointplot(errorbar=iqr,
                  data=subplot_df, x="percentages", y="test_mse",
                  join=False,
                  markers="d", scale=.5, errwidth=0.5, estimator=np.median)

    fs_np = subplot_df["percentages"].max()
    plt.axhline(subplot_df[subplot_df["percentages"] == fs_np]["test_mse"].median(),
                zorder=-3, linewidth=0.5, linestyle='--')

    fd_np = subplot_df["percentages"].min()
    plt.axhline(subplot_df[subplot_df["percentages"] == fd_np]["test_mse"].median(),
                zorder=-1, linewidth=0.5, linestyle='--')

def read_swag_data(path):

    df = pd.DataFrame()
    nll_test = []
    mse_test = []
    percentages = []
    runs = []
    for p in os.listdir(path):
        p = os.path.join(path, p)
        results = pickle.load(open(p, 'rb'))
        nll_test += results['test_nll']
        mse_test += results['test_mse']
        percentages += results['percentages']
        runs += [int(os.path.basename(p).split("_")[-1].split(".")[0])]*len(results['percentages'])

    df['test_nll'] = nll_test
    df['test_mse'] = mse_test
    df['percentages'] = percentages
    df['run'] = runs

    return df


if __name__ == '__main__':
    path = r'C:\Users\45292\Documents\Master\Swag Simple\UCI\Results\boston_models' \
           r''

    df = read_swag_data(path)
    plt.subplots(1,1)
    create_plot_from_df(df)
    plt.show()
    breakpoint()