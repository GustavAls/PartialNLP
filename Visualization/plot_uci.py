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
from plot_utils_comb import PlotHelper
from sklearn.linear_model import LinearRegression
import uncertainty_toolbox as uct
from plot_number_of_parameters import *


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

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_percentages(df, errorbar_func, estimator=None, ax=None, y='Laplace_nll', num_color_schemes=0, map=True):
    # cmap = get_cmap(20)
    # color_num = np.random.randint(2, 18)
    # color_map = mpl.colormaps['Set1']

    point_err_color_dict = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:red'}
    point_err_color = point_err_color_dict[num_color_schemes]
    # plt.tight_layout()
    sns.pointplot(errorbar=errorbar_func,
                  data=df, x="percentages", y=y,
                  join=False,
                  capsize=.30,
                  markers="d",
                  scale=1.0,
                  err_kws={'linewidth': 0.7}, estimator=estimator,
                  color=point_err_color,
                  label=y.split('_')[0],
                  ax=ax)

    try:
        nll_array = np.array(df[y]).reshape((-1, 10))
    except:
        nll_array = np.array(df[y]).reshape((-1, 9))

    estimated = estimator(nll_array, axis=0)
    if map:
        map_val = estimated[0]
    fully_stochastic_val = estimated[-1]

    # Adjust layout
def plot_regression_with_uncertainty(ax, metrics, estimator = np.median, label = "", color = None):
    percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
    # percentages = percentages[4:]
    # metrics = metrics[:, 4:]
    if metrics.shape[-1] != len(percentages):
        metrics = metrics[:, 1:]
    estimated = estimator(metrics, 0)
    if estimator.__name__ == 'median':
        lower, upper = np.percentile(metrics, q=(25, 75), axis = 0)
    elif estimator.__name__ == 'mean':
        std_ = np.std(metrics, 0)
        lower, upper = (estimated - 1.96 * std_, estimated + 1.96 * std_)

    ax.plot(percentages, estimated, label = label, color = color, linestyle = 'dashed')
    # ax.fill_between(percentages, lower, upper, color = color, alpha = 0.3)



def plot_errorbar_percentages(df, errorbar_func, estimator=None, ax=None, y='Laplace_nll', color_scheme_1=True, map=True):
    map_color = 'tab:red'
    stochastic_color = 'tab:green'
    point_err_color = 'tab:blue' if color_scheme_1 else 'tab:orange'

    sns.pointplot(errorbar=errorbar_func,
                  data=df, x="percentages", y=y,
                  join=False,
                  capsize=.30,
                  markers="d",
                  scale=1.0,
                  err_kws={'linewidth': 0.7}, estimator=estimator,
                  color=point_err_color,
                  label=y.split('_')[0],
                  ax=ax)

    if df[y].shape[0] == 45:
        nll_array = np.array(df[y]).reshape((15, 3))
    else:
        try:
            nll_array = np.array(df[y]).reshape((-1, 10))
        except:
            nll_array = np.array(df[y]).reshape((-1, 9))

    estimated = estimator(nll_array, axis=0)
    map_val = estimated[0]
    if map:
        ax.axhline(y=map_val, linestyle='--', linewidth=1, alpha=0.7,
                   color=map_color, label='MAP' if not color_scheme_1 else '_nolegend_')
    fully_stochastic_val = estimated[-1]
    ax.axhline(y=fully_stochastic_val, linestyle='--', linewidth=1, alpha=0.7,
               color=stochastic_color, label='100% Stochastic' if not color_scheme_1 else '_nolegend_')

    # Adjust layout
    # plt.tight_layout()


def plot_estimator_multi_dataset(df, errorbar_func, estimator=None, ax=None, data_name=None, show=True, map=True):
    method_names = []
    np.random.seed(42)
    color_scheme_num = 0

    for i, key in enumerate(df):
        if 'percentages' not in key:
            method_names.append(key.split('_')[0])
            plot_percentages(df=df,
                             errorbar_func=errorbar_func,
                             estimator=estimator,
                             y=key,
                             ax=ax,
                             num_color_schemes=color_scheme_num, map=map)
            color_scheme_num += 1

    title = " & ".join(method_names) + " - " + estimator.__name__ + " - " + data_name
    ax.set_title(label=title, pad=0)
    ax.set_xlabel("Percentages")
    ax.set_ylabel("nll")
    ax.legend()

    # Customize the grid appearance
    # ax.grid(axis='y', linestyle='-', alpha=0.3)
    # ax.grid(b=True, which='major', color='w', linewidth=1.0)
    # Set the Seaborn style
    # plt.savefig(os.path.join(os.getcwd(), title + '.png'))
    if show:
        plt.show(block=False)

def plot_estimator(df, errorbar_func, estimator=None, ax=None, data_name=None, show=True, map=True):
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
                                      color_scheme_1=color_scheme_1, map=map)
            color_scheme_1 = not color_scheme_1

    method_names[0] = 'KFAC Laplace'
    title = " & ".join(method_names) + " - " + estimator.__name__ + " - " + data_name
    ax.set_title(label=title, pad=0)
    # Set labels and legend
    ax.set_xlabel("Num. Modules")

    if df[key].shape[0] == 45:
        nll_array = np.array(df[key]).reshape((15, 3))
    else:
        try:
            nll_array = np.array(df[key]).reshape((-1, 10))
        except:
            nll_array = np.array(df[key]).reshape((-1, 9))

    estimated = estimator(nll_array, axis=0)
    map_val = estimated[0]
    if map:
        ax.axhline(y=map_val, linestyle='--', linewidth=1, alpha=0.7,
                   color=map_color, label='MAP' if not color_scheme_1 else '_nolegend_')
    fully_stochastic_val = estimated[-1]
    ax.axhline(y=fully_stochastic_val, linestyle='--', linewidth=1, alpha=0.7,
               color=stochastic_color, label='100% Stochastic' if not color_scheme_1 else '_nolegend_')
    ax.axhline(y=2.51,linestyle='--', linewidth=1, alpha=0.7,
               color='tab:red', label='GGN Laplace 5 pct')
    if show:
        plt.show(block=False)


def plot_partial_percentages(percentages, res, data_name=None, df=None, num_runs=15, ax=None, show=True,
                             multidataset = False, map=True, is_ll=True):
    if df is None:
        df = pd.DataFrame()
        df['percentages'] = num_runs * percentages
        for key, val in res.items():
            df[key + '_nll'] = [(-1) * ll if is_ll else ll for ll in val.flatten()]

    if ax is None:
        fig1, ax1 = plt.subplots(1, 1)
    else:
        ax1 = ax[0]


    if multidataset:
        plot_estimator_multi_dataset(df=df, errorbar_func=lambda x: np.percentile(x, (25, 75)),
                   estimator=np.median, ax=ax1, data_name=data_name,
                   show=show, map=map)
    else:
        plot_estimator(df=df, errorbar_func=lambda x: np.percentile(x, (25, 75)),
                       estimator=np.median, ax=ax1, data_name=data_name,
                       show=show, map=map)

    # Plot mean
    if ax is None:
        fig2, ax2 = plt.subplots(1, 1)
    else:
        ax2 = ax[1]

    if multidataset:
        plot_estimator_multi_dataset(df=df, errorbar_func=('ci', 95), estimator=np.mean,
                                     ax=ax2, data_name=data_name, show=show, map=map)
    else:
        plot_estimator(df=df, errorbar_func=('ci', 95), estimator=np.mean, ax=ax2,
                       data_name=data_name, show=show, map=map)



def read_data_swag_la(path, include_map=True):
    percentages =   []
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


def plot_scatter(predictions, labels, data_name=None, method_name=None, df=None,
                 epsilon=0.1, minmax=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    errorbar_func = lambda x: np.percentile(x, (2.5, 97.5))

    df = pd.DataFrame()

    # sns.pointplot(errorbar=errorbar_func,
    #               data=df, x="labels", y='preds',
    #               join=False,
    #               capsize= 0.15,
    #               markers="d",
    #               scale=1.0,
    #               err_kws={'linewidth': 0.7}, estimator=np.mean,
    #               color='tab:orange',
    #               label= 'Predictions and labels',
    #               ax=ax)

    point_color = 'tab:orange'
    ax.scatter(labels, predictions.mean(-1), marker='d', s=20,
               label='Predictions and labels', color=point_color)
    errs = np.percentile(predictions, (2.5, 97.5), axis=1)
    errs = np.abs((errs - predictions.mean(-1)))

    ax.errorbar(labels, predictions.mean(-1), yerr=errs, fmt='none', capsize=4,
                color=point_color, alpha=0.7, linewidth=0.7)

    if minmax is None:
        minimum = min((labels.min(), np.min(predictions.mean(-1) + errs),
                       np.min(predictions.mean(-1) - errs))) - epsilon
        maximum = max((labels.max(), np.max(predictions.mean(-1) + errs),
                       np.max(predictions.mean(-1) - errs))) + epsilon
    elif isinstance(minmax, (tuple, list)):
        minimum, maximum = minmax

    ax.plot(np.linspace(minimum, maximum, 200), np.linspace(minimum, maximum, 200),
            linestyle='--', linewidth=1.4,
            color='tab:green', label='Ideal curve'
            )

    lin_mod = LinearRegression().fit(labels[:, None], predictions.mean(-1)[:, None])

    preds = lin_mod.predict(np.linspace(minimum, maximum, 200)[:, None])
    ax.plot(np.linspace(minimum, maximum, 200), preds,
            linestyle='--', linewidth=1.4,
            color='tab:red', label='Linear Trend'
            )

    ax.set_ylim(ymin=minimum, ymax=maximum)
    ax.set_xlim(xmin=minimum, xmax=maximum)
    ax.legend()

    # sns.pointplot(errorbar=errorbar_func,
    #               data=df, x="labels", y='labels',
    #               join=False,
    #               capsize= 0.15,
    #               markers="d",
    #               scale=1.0,
    #               color='tab:blue',
    #               label= 'Ideal Predictions',
    #               ax=ax)
    #
    # sns.lineplot(data=df, x="labels", y='labels',
    #               color='tab:blue',
    #               label= 'Ideal predictions',
    #               ax=ax)

    # ax.set_xticks(labels[::5])

    # ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Predictions')
    data_name = '' if data_name is None else ", " + data_name
    title = f'Preds and labels for {method_name}' + data_name
    ax.set_title(title)
    return ax


def plot_calibration(predictions, labels, pred_var=None, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    predictions_mean = predictions.mean(-1)
    predictions_std = predictions.std(-1)

    uct.plot_calibration(predictions_mean, predictions_std, labels, ax=ax, curve_label=label, show_miscalib=False)


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
                    test_ll_hmc[perc].append(pcl[perc]['test_ll_theirs'])
        else:
            test_ll_vi.append(pcl['test_ll_theirs'])
            percentages.append(pcl['percentiles'])

    percentages = percentages[-1]
    test_ll_vi = np.array(test_ll_vi)
    test_ll_hmc = dict_to_nparray(test_ll_hmc)

    return percentages, test_ll_vi, test_ll_hmc


def dict_to_nparray(dic):
    percentages, data = [], []
    for key, val in dic.items():
        percentages.append([0] * len(val) if 'map' in key else [int(key)] * len(val))
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


def set_legends_to_plot(ax, criteria='Ideal', max_count=1):
    lines, labels = [], []
    counter = 0

    holder = []
    cntd = 0

    for idx, child in enumerate(ax._children):
        if isinstance(child, mpl.collections.PolyCollection):
            holder.append(ax._children[cntd: idx + 1])
            cntd = idx + 1

    for hold in holder:
        fill_between = hold[-1]
        colors = fill_between.get_facecolors().copy()
        colors[0, -1] = 1
        for child in hold[:-1]:
            child.set_color(colors)

    ideal_color = '#ff7f0e'
    for line in ax.lines:
        if criteria not in line._label or counter < max_count:
            if criteria in line._label:
                line.set_color(ideal_color)
            lines.append(line)
            labels.append(line._label)
            counter += 1

    ax.legend(lines, labels)


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


def calculate_max_and_min_for_predictions(ph, run_type, epsilon, idx=0, laplace=False):
    predictions, labels = ph.get_predictions_and_labels_for_percentage('100', idx, run_type, laplace=laplace)

    errs = np.percentile(predictions, (2.5, 97.5), axis=1)
    errs = np.abs((errs - predictions.mean(-1)))

    minimum = min((labels.min(), np.min(predictions.mean(-1) + errs),
                   np.min(predictions.mean(-1) - errs))) - epsilon
    maximum = max((labels.max(), np.max(predictions.mean(-1) + errs),
                   np.max(predictions.mean(-1) - errs))) + epsilon

    return minimum, maximum


def calculate_minmax_for_laplace(ph, run_type, epsilon, idx):
    predictions, labels = ph.get_predictions_and_labels_for_percentage('100', idx, run_type, laplace=True)
    fvar = predictions[1]
    errs = np.sqrt(fvar) * 1.96

    minimum = min((labels.min(), np.min(predictions + errs),
                   np.min(predictions - errs))) - epsilon
    maximum = max((labels.max(), np.max(predictions + errs),
                   np.max(predictions - errs))) + epsilon
    return minimum, maximum


def set_dataset(path1, path2):
    correct_ = pickle.load(open(path1, 'rb'))
    to_be_changed = pickle.load(open(path2, 'rb'))

    to_be_changed['dataset_'] = to_be_changed['dataset']
    to_be_changed['dataset'] = correct_['dataset']

    with open(path2, 'wb') as handle:
        pickle.dump(to_be_changed, handle, protocol=pickle.HIGHEST_PROTOCOL)


def change_datasets(path):
    laplace, swag = [], []
    for p in os.listdir(path):
        if 'laplace' in p:
            laplace.append(
                (int(p.split("_")[-1].split(".")[0]), os.path.join(path, p))
            )
        if 'swag' in p:
            swag.append(
                (int(p.split("_")[-1].split(".")[0]), os.path.join(path, p))
            )

    for (idx, p_la), (i, p_swa) in zip(sorted(laplace), sorted(swag)):
        if i == idx:
            set_dataset(p_la, p_swa)


class PlotFunctionHolder:

    def __init__(self, la_swa_path = "", vi_hmc_path = "", eval_method = 'nll', calculate=True, show=True,
                 save_path = None, la_swa_path_rand = "", vi_hmc_path_rand = "", la_var_path = ""):
        self.la_swa_path = la_swa_path
        self.vi_hmc_path = vi_hmc_path
        self.la_var_path = la_var_path
        self.la_swa_path_rand = la_swa_path_rand
        self.vi_hmc_path_rand = vi_hmc_path_rand
        self.calculate = calculate
        self.save_path = save_path

        self.plot_helper_la_swa = PlotHelper(self.la_swa_path, eval_method=eval_method, calculate=calculate)
        self.plot_helper_vi_hmc = PlotHelper(self.vi_hmc_path, eval_method=eval_method, calculate=calculate)

        if self.la_var_path != "":
            self.plot_helper_la_var = PlotHelper(self.la_var_path, eval_method=eval_method, calculate=calculate)
        if self.la_swa_path_rand != "":
            self.plot_helper_la_swa_rand = PlotHelper(self.la_swa_path_rand, eval_method=eval_method, calculate=calculate)
        if self.vi_hmc_path_rand != "":
            self.plot_helper_vi_hmc_rand = PlotHelper(self.vi_hmc_path_rand, eval_method=eval_method, calculate=calculate)

        self.folder_name_to_data_name = {
            'energy': 'Energy', 'boston': 'Boston', 'yacht': 'Yacht'
        }

        self.eval_methods_to_names = {
            'mse': 'MSE', 'nll_glm': 'NLL','nll': 'NLL', 'glm_nll': 'NLL', 'elpd': 'ELPD', 'calib': 'Calibration',
            'sqrt': 'ELPD', 'test_ll_homoscedastic' : 'NLL Homoscedastic',
            'prior_precision': 'Prior Precision'}

        self.percentages = [0, 1, 2, 5, 8, 14, 23, 37, 61, 100]

        self.show_ = show

    def set_eval_method(self, new_eval_method):
        self.plot_helper_vi_hmc.eval_method = new_eval_method
        self.plot_helper_la_swa.eval_method = new_eval_method

    def find_data_name(self, path):
        data_name = os.path.basename(path).split("_")[0]

        return self.folder_name_to_data_name[data_name]

    def get_eval_method_name(self, eval_method):
        return self.eval_methods_to_names[eval_method]

    def show(self):
        if self.show_:
            plt.show()

    def plot_number_of_parameters(self, save_path):

        results = simulate_n_times()
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)
        plot_with_error_bars(percentile_mat=results,
                             path=os.path.join(save_path, 'num_params_w_LSVH.pdf'), show_big=True, ax = ax1)
        self.adjust_yscale(ax1)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                  ncol=1, fancybox=True, shadow=True)
        fig1.tight_layout()
        fig1.savefig(os.path.join(save_path, 'num_params_w_LSVH.pdf'), format = 'pdf')

        self.show()
        plot_with_error_bars(percentile_mat=results,
                             path=os.path.join(save_path, 'num_params_m_LSVH.pdf'), show_big=False, ax = ax2)
        self.adjust_yscale(ax2)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                   ncol=1, fancybox=True, shadow=True)
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_path, 'num_params_m_LSVH.pdf'))

        self.show()

    def plot_prior_laplace(self, model_paths=None, save_path = None):

        potential_paths = ['energy_models', 'yacht_models', 'boston_models']
        org_calculate = self.plot_helper_la_swa.calculate
        setattr(self.plot_helper_la_swa, 'calculate', False)
        setattr(self.plot_helper_la_swa, 'eval_method', 'prior_precision')

        prior_precision_la = self.plot_helper_la_swa.run_for_dataset(criteria='laplace', laplace=True)
        prior_precision_la = np.array(prior_precision_la)[:, 1: ]
        name = self.find_data_name(self.la_swa_path)
        res = {name: prior_precision_la}
        if model_paths is None:
            model_paths = sorted(list(set(potential_paths)-set(os.path.basename(self.la_swa_path))))
        res_tmp = self.get_criteria_between_datasets_la_swa(model_paths, criteria='laplace', laplace=True)
        for key, val in res_tmp.items():
            res[key] = val[:, 1:]

        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)

        percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
        plot_partial_percentages(percentages=percentages,
                                 res=res,
                                 data_name='Laplace Prior Precision',
                                 num_runs=len(prior_precision_la),
                                 ax=[ax1, ax2], show=False, multidataset=True)

        ax1.set_ylabel(self.eval_methods_to_names['prior_precision'])
        ax2.set_ylabel(self.eval_methods_to_names['prior_precision'])
        fig1.tight_layout()
        fig2.tight_layout()

        save_path = save_path if save_path is not None else self.save_path
        if save_path is not None:
            fig1.savefig(os.path.join(save_path, 'Laplace_prior_precision_median.pdf'), format='pdf')
            fig2.savefig(os.path.join(save_path, 'Laplace_prior_precision_mean.pdf'), format='pdf')

        self.show()

    def get_criteria_between_datasets_la_swa(self, potential_paths, criteria, laplace = False):

        results = {}
        if all([os.path.isdir(path) for path in potential_paths]):
            paths = potential_paths
        else:
            paths = [os.path.join(os.path.dirname(self.la_swa_path), pa) for pa in potential_paths]
            paths = [p for p in paths if os.path.isdir(p)]

        for path in paths:
            new_plot_helper = PlotHelper(path, self.plot_helper_la_swa.eval_method, self.plot_helper_la_swa.calculate)
            res = new_plot_helper.run_for_dataset(criteria=criteria, laplace=laplace)
            name = self.find_data_name(path)
            results[name] = np.array(res)

        return results

    def plot_pred_labels_la_swa(self, percentages=('1', '100')):

        data_name = self.find_data_name(self.la_swa_path)
        for method, criteria in  [('Laplace', 'laplace'), ('SWAG', 'swag')]:
            minmax = calculate_max_and_min_for_predictions(self.plot_helper_la_swa, run_type=criteria, epsilon=0.1, idx=4,
                                                           laplace=True if 'laplace' in criteria else False)
            for percentage in percentages:
                fig, ax = plt.subplots(1, 1)
                predictions, labels = self.plot_helper_la_swa.get_predictions_and_labels_for_percentage(
                    percentage, 4, criteria, laplace=True if 'laplace' in criteria else False)
                plot_scatter(predictions, labels, data_name=data_name, minmax=minmax,
                             method_name=f"{method}, {percentage} stoch", ax=ax)
                save_path = self.save_path
                if save_path is not None:
                    fig.savefig(os.path.join(save_path, f'Preds_labels_{data_name}_{method}_perc_{percentage}.pdf'), format='pdf')
                    self.show()

    def plot_pred_labels_vi_hmc(self, percentages=('1', '100')):

        data_name = self.find_data_name(self.la_swa_path)
        for method, critera in [('VI', 'vi_run'), ('HMC', 'hmc')]:

            minmax = calculate_max_and_min_for_predictions(
                self.plot_helper_vi_hmc, run_type=critera, epsilon=0.1, idx=4)
            for percentage in percentages:
                fig, ax = plt.subplots(1, 1)
                predictions, labels = self.plot_helper_vi_hmc.get_predictions_and_labels_for_percentage(
                    percentage, 4, critera)
                plot_scatter(predictions, labels, data_name=data_name, minmax=minmax,
                             method_name=f"{method}, {percentage} stoch", ax=ax)
                save_path = self.save_path
                if save_path is not None:
                    fig.savefig(os.path.join(save_path, f'Preds_labels_{data_name}_{method}_perc_{percentage}.pdf'), format='pdf')
                self.show()

    def plot_pred_labels_node_based(self, percentages=('1', '100')):

        data_name = self.find_data_name(self.la_swa_path)

        for method, critera in [('Additive', 'add'), ('Multiplicative', 'node_run')]:

            minmax = calculate_max_and_min_for_predictions(
                self.plot_helper_vi_hmc, run_type=critera, epsilon=0.1, idx=4)
            for percentage in percentages:
                fig, ax = plt.subplots(1, 1)
                predictions, labels = self.plot_helper_vi_hmc.get_predictions_and_labels_for_percentage(
                    percentage, 4, critera)
                plot_scatter(predictions, labels, data_name=data_name, minmax=minmax,
                             method_name=f"{method}, {percentage} stoch", ax=ax)
                if save_path is not None:
                    fig.savefig(os.path.join(save_path, f'Preds_labels_{data_name}_{method}_perc_{percentage}.pdf'), format='pdf')
                self.show()

    def set_save_path(self, path):
        self.save_path = path

    def plot_partial_percentages_nodes(self, save_path = None, map=True):

        metrics_mul = self.plot_helper_vi_hmc.run_for_dataset(criteria='node_run', map=map)
        metrics_add = self.plot_helper_vi_hmc.run_for_dataset(criteria='add', map=map)
        percentages = [0, 1, 2, 5, 8, 14, 23, 37, 61, 100] if map else [1, 2, 5, 8, 14, 23, 37, 61, 100]

        data_name = self.find_data_name(self.vi_hmc_path) + " " + self.get_eval_method_name(
            self.plot_helper_vi_hmc.eval_method)

        if (l1 := len(metrics_add)) != (l2 := len(metrics_mul)):
            min_ = min((l1, l2))
            metrics_mul = metrics_mul[:min_]
            metrics_add = metrics_add[:min_]
        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)
        plot_partial_percentages(percentages=percentages,
                                 res={'Additive': np.array(metrics_add), 'Multiplicative': np.array(metrics_mul)},
                                 data_name=data_name,
                                 num_runs=len(metrics_mul),
                                 ax=[ax1,ax2], show=False, map=map,
                                 is_ll=True if self.plot_helper_vi_hmc.eval_method == 'nll' else False)

        ylabel = self.eval_methods_to_names[self.plot_helper_vi_hmc.eval_method]
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

        self.set_bounds_and_layout((np.array(metrics_add), np.array(metrics_mul)), np.median, fig1, ax1)
        self.set_bounds_and_layout((np.array(metrics_add), np.array(metrics_mul)), np.mean, fig2, ax2)
        save_path = save_path if save_path is not None else self.save_path
        if save_path is not None:
            fig1.savefig(os.path.join(save_path, f'node_based_{ylabel}_{data_name}_{"no map" if not map else str()}_median.pdf'), format='pdf')
            fig2.savefig(os.path.join(save_path, f'node_based_{ylabel}_{data_name}_{"no map" if not map else str()}_mean.pdf'), format='pdf')

        self.show()

    def plot_partial_percentages_node_mult(self, save_path = None, map=True):

        metrics_mul = self.plot_helper_vi_hmc.run_for_dataset(criteria='node_run', map=map)
        metrics_mul_max = self.plot_helper_vi_hmc.run_for_dataset(criteria='max', map=map)
        percentages = [0, 1, 2, 5, 8, 14, 23, 37, 61, 100] if map else [1, 2, 5, 8, 14, 23, 37, 61, 100]

        data_name = self.find_data_name(self.vi_hmc_path) + " " + self.get_eval_method_name(
            self.plot_helper_vi_hmc.eval_method)

        if (l1 := len(metrics_mul_max)) != (l2 := len(metrics_mul)):
            min_ = min((l1, l2))
            metrics_mul = metrics_mul[:min_]
            metrics_mul_max = metrics_mul_max[:min_]
        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)
        plot_partial_percentages(percentages=percentages,
                                 res={'Multi max': np.array(metrics_mul_max), 'Multi rand': np.array(metrics_mul)},
                                 data_name=data_name,
                                 num_runs=len(metrics_mul),
                                 ax=[ax1,ax2], show=False, map=map,
                                 is_ll=True if self.plot_helper_vi_hmc.eval_method == 'nll' else False)

        ylabel = self.eval_methods_to_names[self.plot_helper_vi_hmc.eval_method]
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

        self.set_bounds_and_layout((np.array(metrics_mul_max), np.array(metrics_mul)), np.median, fig1, ax1)
        self.set_bounds_and_layout((np.array(metrics_mul_max), np.array(metrics_mul)), np.mean, fig2, ax2)
        save_path = save_path if save_path is not None else self.save_path
        if save_path is not None:
            fig1.savefig(os.path.join(save_path, f'nb_multi_{ylabel}_{data_name}_{"no map" if not map else str()}_median.pdf'), format='pdf')
            fig2.savefig(os.path.join(save_path, f'nb_multi_{ylabel}_{data_name}_{"no map" if not map else str()}_mean.pdf'), format='pdf')

        self.show()

    def ensure_same_length_and_get_name(self, metrics, data_path, eval_method):

        min_ = min((len(met) for met in metrics))
        metrics = [met[:min_] for met in metrics]

        data_name = self.find_data_name(data_path) + " " + self.get_eval_method_name(eval_method)
        return metrics, data_name

    # Sharma HMC plot
    def plot_partial_percentages_hmc_homoscedastic_nll(self, save_path = None):
        metrics_hmc = self.plot_helper_vi_hmc.run_for_dataset(criteria='hmc')
        percentages = [0, 1, 2, 5, 8, 14, 23, 37, 61, 100]

        data_name = self.find_data_name(self.vi_hmc_path) + " " + self.get_eval_method_name(
            self.plot_helper_vi_hmc.eval_method)

        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)
        plot_partial_percentages(percentages=percentages,
                                 res={'HMC': np.array(metrics_hmc)},
                                 data_name=data_name,
                                 num_runs=len(metrics_hmc),
                                 ax=[ax1,ax2], show=False, map=True)

        save_path = save_path if save_path is not None else self.save_path

        ylabel = self.eval_methods_to_names[self.plot_helper_la_swa.eval_method]
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)
        self.set_bounds_and_layout((np.array(metrics_hmc), np.array(metrics_hmc)), np.median, fig1, ax1)
        self.set_bounds_and_layout((np.array(metrics_hmc), np.array(metrics_hmc)), np.mean, fig2, ax2)

        if save_path is not None:
            fig1.savefig(
                os.path.join(save_path, f'hmc_homoscedastic_{data_name}_median.pdf'),
                format='pdf')
            fig2.savefig(
                os.path.join(save_path, f'hmc_homoscedastic_{data_name}_mean.pdf'),
                format='pdf')
        self.show()

    def plot_partial_percentages_vi_hmc(self, save_path = None, map=True):
        metrics_vi = self.plot_helper_vi_hmc.run_for_dataset(criteria='vi_run', map=map)
        metrics_hmc = self.plot_helper_vi_hmc.run_for_dataset(criteria='hmc', map=map)
        percentages = [0, 1, 2, 5, 8, 14, 23, 37, 61, 100] if map else [1, 2, 5, 8, 14, 23, 37, 61, 100]

        (metrics_vi, metrics_hmc), data_name = self.ensure_same_length_and_get_name(
            [metrics_vi, metrics_hmc], self.vi_hmc_path, self.plot_helper_vi_hmc.eval_method
        )

        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)

        plot_partial_percentages(percentages=percentages,
                                 res={'VI': np.array(metrics_vi), 'HMC': np.array(metrics_hmc)},
                                 data_name=data_name,
                                 num_runs=len(metrics_vi),
                                 ax=[ax1, ax2], show=False, map=map,
                                 is_ll=True if self.plot_helper_vi_hmc.eval_method == 'nll' else False)

        ylabel = self.eval_methods_to_names[self.plot_helper_la_swa.eval_method]
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)
        self.set_bounds_and_layout((np.array(metrics_vi), np.array(metrics_hmc)), np.median, fig1, ax1)
        self.set_bounds_and_layout((np.array(metrics_vi), np.array(metrics_hmc)), np.mean, fig2, ax2)

        save_path = save_path if save_path is not None else self.save_path

        if save_path is not None:
            fig1.savefig(os.path.join(save_path, f'vi_hmc_{ylabel}_{data_name}_{"no map" if not map else str()}_median.pdf'), format='pdf')
            fig2.savefig(os.path.join(save_path, f'vi_hmc_{ylabel}_{data_name}_{"no map" if not map else str()}_mean.pdf'), format='pdf')

        self.show()


    def set_bounds_and_layout(self, metrics, estimator, fig, ax):

        self.calculate_new_y_bounds_from_data(metrics, ax, estimator)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                   ncol=2, fancybox=True, shadow=True)
        fig.tight_layout()

    def calculate_new_y_bounds(self, ax):
        y_bounds = ax.get_ybound()
        diff = y_bounds[1] - y_bounds[0]
        new_bounds = (y_bounds[0], y_bounds[1] + diff/5)
        ax.set_ylim(new_bounds)

    def calculate_new_y_bounds_from_data(self, metrics, ax, estimator = np.median):
        if isinstance(metrics, (list, tuple)):
            met = metrics[0][:, 1:-1]
            for m in metrics[1:]:
                met = np.concatenate((met, m[:, 1:-1]), -1)
        else:
            met = metrics[:, 1:-1]

        ylims = ax.get_ybound()
        if np.all(estimator(np.abs(met), axis=0) < ylims[1]-(ylims[1] - ylims[0])/5):
            return None

        self.calculate_new_y_bounds(ax)

    def plot_partial_percentages_kron(self, save_path = None):
        metrics_la = self.plot_helper_la_swa.run_for_dataset(criteria='laplace', laplace=True, map=False)
        num_modules = [1,2,3]

        ylabel = self.eval_methods_to_names[self.plot_helper_la_swa.eval_method]
        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)
        data_name = data_name = self.find_data_name(self.la_swa_path) + " " + self.get_eval_method_name(
            self.plot_helper_la_swa.eval_method
        )
        plot_partial_percentages(percentages=num_modules,
                                 res={'Laplace': np.array(metrics_la)},
                                 data_name=data_name,
                                 num_runs=len(metrics_la),
                                 ax=[ax1, ax2], show=False, map=False,
                                 is_ll=True if self.plot_helper_vi_hmc.eval_method in ['nll', 'nll_glm'] else False)

        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                  ncol=1, fancybox=True, shadow=True)
        ax2.legend()
        save_path = save_path if save_path is not None else self.save_path

        if save_path is not None:
            fig1.savefig(
                os.path.join(save_path, f'la_swa_{ylabel}_{data_name}_{"no map" if not False else str()}_median.pdf'),
                format='pdf')
            fig2.savefig(
                os.path.join(save_path, f'la_swa_{ylabel}_{data_name}_{"no map" if not False else str()}_mean.pdf'),
                format='pdf')

        self.show()

    def plot_partial_percentages_la_swa(self, save_path = None, map=True):
        metrics_la = self.plot_helper_la_swa.run_for_dataset(criteria='laplace', laplace=True, map=map)
        metrics_swa = self.plot_helper_la_swa.run_for_dataset(criteria='swag', laplace=False, map=map)
        percentages = [0, 1, 2, 5, 8, 14, 23, 37, 61, 100] if map else [1, 2, 5, 8, 14, 23, 37, 61, 100]
        (metrics_la, metrics_swa), data_name = self.ensure_same_length_and_get_name(
            [metrics_la, metrics_swa], self.la_swa_path, self.plot_helper_la_swa.eval_method
        )

        ylabel = self.eval_methods_to_names[self.plot_helper_la_swa.eval_method]
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)

        plot_partial_percentages(percentages=percentages,
                                 res={'SWAG': np.array(metrics_swa), 'Laplace': np.array(metrics_la)},
                                 data_name=data_name,
                                 num_runs=len(metrics_la),
                                 ax=[ax1,ax2], show=False, map=map,
                                 is_ll=True if self.plot_helper_vi_hmc.eval_method == 'nll' else False)


        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

        self.set_bounds_and_layout((np.array(metrics_swa), np.array(metrics_la)), np.median, fig1, ax1)
        self.set_bounds_and_layout((np.array(metrics_swa), np.array(metrics_la)), np.mean, fig2, ax2)

        save_path = save_path if save_path is not None else self.save_path

        if save_path is not None:
            fig1.savefig(os.path.join(save_path, f'la_swa_{ylabel}_{data_name}_{"no map" if not map else str()}_median.pdf'), format='pdf')
            fig2.savefig(os.path.join(save_path, f'la_swa_{ylabel}_{data_name}_{"no map" if not map else str()}_mean.pdf'), format='pdf')

        self.show()


    def plot_partial_percentages_la_vs_la_var_select(self, save_path = None, map=True):
        metrics_la = self.plot_helper_la_swa.run_for_dataset(criteria='laplace', laplace=True, map=map)
        metrics_la_var = self.plot_helper_la_var.run_for_dataset(criteria='laplace', laplace=True, map=map)
        percentages = [0, 1, 2, 5, 8, 14, 23, 37, 61, 100] if map else [1, 2, 5, 8, 14, 23, 37, 61, 100]
        (metrics_la, metrics_la_var), data_name = self.ensure_same_length_and_get_name(
            [metrics_la, metrics_la_var], self.la_var_path, self.plot_helper_la_swa.eval_method
        )

        ylabel = self.eval_methods_to_names[self.plot_helper_la_swa.eval_method]
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)

        plot_partial_percentages(percentages=percentages,
                                 res={'Laplace $\mu$': np.array(metrics_la), 'Laplace $\sigma^2$': np.array(metrics_la_var)},
                                 data_name=data_name,
                                 num_runs=len(metrics_la),
                                 ax=[ax1,ax2], show=False, map=map,
                                 is_ll=True if self.plot_helper_vi_hmc.eval_method == 'nll' else False)


        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

        self.set_bounds_and_layout((np.array(metrics_la_var), np.array(metrics_la)), np.median, fig1, ax1)
        self.set_bounds_and_layout((np.array(metrics_la_var), np.array(metrics_la)), np.mean, fig2, ax2)

        save_path = save_path if save_path is not None else self.save_path

        if save_path is not None:
            fig1.savefig(os.path.join(save_path, f'la_la_var_{ylabel}_{data_name}_{"no map" if not map else str()}_median.pdf'), format='pdf')
            fig2.savefig(os.path.join(save_path, f'la_la_var_{ylabel}_{data_name}_{"no map" if not map else str()}_mean.pdf'), format='pdf')

        self.show()


    def plot_partial_percentages_rand_vs_max(self, save_path = None, map=True, criteria='laplace'):
        laplace = True if criteria == 'laplace' else False
        if criteria == 'laplace' or criteria == 'swag':
            metrics_max = self.plot_helper_la_swa.run_for_dataset(criteria=criteria, laplace=laplace, map=map)
            metrics_rand = self.plot_helper_la_swa_rand.run_for_dataset(criteria=criteria, laplace=laplace, map=map)
            for i in range(len(metrics_rand)):
                metrics_rand[i][-1] = metrics_max[i][-1]
        else:
            metrics_max = self.plot_helper_vi_hmc.run_for_dataset(criteria=criteria, laplace=laplace, map=map)
            metrics_rand = self.plot_helper_vi_hmc_rand.run_for_dataset(criteria=criteria, laplace=laplace, map=map)
        percentages = [0, 1, 2, 5, 8, 14, 23, 37, 61, 100] if map else [1, 2, 5, 8, 14, 23, 37, 61, 100]
        data_name = self.find_data_name(self.vi_hmc_path) + " " + self.get_eval_method_name(
            self.plot_helper_vi_hmc.eval_method)

        ylabel = self.eval_methods_to_names[self.plot_helper_la_swa.eval_method]
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)

        if criteria == 'laplace':
            plot_criteria = criteria.capitalize()
        elif criteria == 'vi_run':
            plot_criteria = 'VI'
        elif criteria == 'node_run':
            plot_criteria = 'Multiplicative'
        elif criteria == 'add':
            plot_criteria = 'Additive'
        else:
            plot_criteria = criteria.upper()
        plot_partial_percentages(percentages=percentages,
                                 res={'Max': np.array(metrics_max), 'Random': np.array(metrics_rand)},
                                 data_name=plot_criteria + " " + data_name,
                                 num_runs=len(metrics_max),
                                 ax=[ax1,ax2], show=False, map=map,
                                 is_ll=True if self.plot_helper_vi_hmc.eval_method == 'nll' else False)

        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

        self.set_bounds_and_layout((np.array(metrics_max), np.array(metrics_rand)), np.median, fig1, ax1)
        self.set_bounds_and_layout((np.array(metrics_max), np.array(metrics_rand)), np.mean, fig2, ax2)

        save_path = save_path if save_path is not None else self.save_path

        if save_path is not None:
            fig1.savefig(os.path.join(save_path, f'maxVsRand_{criteria}_{ylabel}_{data_name}_{"no map" if not map else str()}_median.pdf'), format='pdf')
            fig2.savefig(os.path.join(save_path, f'maxVsRand_{criteria}_{ylabel}_{data_name}_{"no map" if not map else str()}_mean.pdf'), format='pdf')

        self.show()

    def adjust_yscale(self, ax):
        fits = True
        for line in ax.lines:
            ylim = ax.get_ybound()
            x, y = line.get_data()
            if ax.get_yscale() == 'log':
                rel = np.log(ylim[1])-np.log(ylim[1]-ylim[0])/5
                overlapping = np.any(np.log(y) > rel)
            else:
                rel = ylim[1] - (ylim[1] - ylim[0]) / 5
                overlapping = np.any(y > rel)

            if overlapping:
                if ax.get_yscale() == 'log':
                    ax.set_ylim((ylim[0], ylim[1]*10))
                else:
                    ax.set_ylim((ylim[0], ylim[1] + rel/8))
                fits = False
        if not fits:
            self.adjust_yscale(ax)

    def find_best_comparable(self):

        metrics_vi = np.array(self.plot_helper_vi_hmc.run_for_dataset('vi_run', laplace=False))
        metrics_add = np.array(self.plot_helper_vi_hmc.run_for_dataset('add', laplace=False))
        metrics_mul = np.array(self.plot_helper_vi_hmc.run_for_dataset('node_run', laplace=False))
        if self.plot_helper_vi_hmc.eval_method in ['nll', 'nll_glm']:
            metrics_add, metrics_mul, metrics_vi = -metrics_add, -metrics_mul, -metrics_vi

        colors = ['tab:red', 'tab:blue', 'tab:green']
        results = {}
        for i, (method, res) in enumerate(
                zip(['VI', 'Add.', 'Mult.'], [metrics_vi,metrics_add, metrics_mul])):
            res = res[:, 1:]
            minimum = np.min(np.median(res, 0))
            best_percentage = self.percentages[np.argmin(np.median(res, 0))]
            results[method] = (minimum, best_percentage, colors[i])

        return results

    def plot_hmc_sample_scaling(self, save_path = None):
        save_path = save_path if save_path is not None else self.save_path
        data_name = self.find_data_name(self.vi_hmc_path)
        sample_results = self.plot_helper_vi_hmc.run_hmc_scaling_for_dataset()

        other_method_results = self.find_best_comparable()

        if self.plot_helper_vi_hmc.eval_method in ['nll', 'nll_glm', 'glm_nll']:
            sample_results = {key: - val for key, val in sample_results.items()}

        map_results = np.array([m[:, 0] for k, m in sample_results.items()])
        palette = sns.color_palette(None, len(sample_results))
        fig, ax = plt.subplots(1,1)
        for idx, (key, val) in enumerate(sample_results.items()):
            plot_regression_with_uncertainty(ax, val, label = f"{int(key)*2}", color = palette[idx])
            print(np.min(val))
        #
        ax.axhline(y=np.median(map_results.mean(0)), linestyle='--', linewidth=2, alpha=0.9,
                   color='tab:orange', label='MAP')
        for key, val in other_method_results.items():
            ax.axhline(y=val[0], linestyle='--', linewidth=2, alpha=0.9,
                       color=val[2], label=key)

        ax.set_xlabel('Percentages')
        y_label = self.eval_methods_to_names[self.plot_helper_vi_hmc.eval_method]
        ax.set_ylabel(y_label)
        ax.set_title(f'Sample Scaling HMC, {data_name} - {y_label}')
        self.adjust_yscale(ax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
                  ncol=4, fancybox=True, shadow=True)
        fig.tight_layout()
        # ax.set_yscale('log')
        if save_path is not None:
            fig.savefig(os.path.join(save_path, f'HMC_scaled_{y_label}_{data_name}.pdf'), format='pdf')
        self.show()

    def plot_calibration_vi_hmc(self, percentages=None, save_path = None):
        if percentages is None:
            percentages = ['1', '5', '61', '100']
        save_path = save_path if save_path is not None else self.save_path
        data_name = self.find_data_name(self.vi_hmc_path)
        num_runs = 15
        for method, criteria in [('VI', 'vi_run'), ('HMC', 'hmc')]:
            fig, ax = plt.subplots(1, 1)
            for run in percentages:
                for idx in range(num_runs):
                    if num_runs == 1:
                        # Choose a different run number than 1
                        run_number = 5
                        predictions, labels = self.plot_helper_vi_hmc.get_predictions_and_labels_for_percentage(
                            percentage=run, idx=run_number, path_name_criteria=criteria, laplace=True if 'laplace' in criteria else False
                        )
                    else:
                        predictions, labels = self.plot_helper_vi_hmc.get_predictions_and_labels_for_percentage(
                            percentage=run, idx=idx, path_name_criteria=criteria,
                            laplace=True if 'laplace' in criteria else False
                        )

                    if idx==0:
                        preds = copy.deepcopy(predictions)
                        labs = copy.deepcopy(labels)
                    else:
                        preds = np.vstack((preds, predictions))
                        labs = np.concatenate((labs, labels))
                plot_calibration(preds, labs.ravel(), ax=ax, label=f'{run} pct')
            set_legends_to_plot(ax)
            ax.set_title(f'Average Calibration {method}, {data_name}')
            fig.tight_layout()
            if save_path is not None:
                fig.savefig(os.path.join(save_path, f'{criteria}_calibration_{data_name}_num_runs_{num_runs}.pdf'), format='pdf',
                            bbox_inches='tight', pad_inches=0.05)

            self.show()

    def plot_calibration_la_swa(self, percentages=None,  save_path = None):

        if percentages is None:
            percentages = ['1', '5', '61', '100']
        save_path = save_path if save_path is not None else self.save_path
        data_name = self.find_data_name(self.la_swa_path)
        num_runs = 15
        for method, criteria in [('Laplace', 'laplace'), ('SWAG', 'swag')]:
            fig, ax = plt.subplots(1, 1)
            for run in percentages:
                for idx in range(num_runs):
                    if num_runs == 1:
                        # Choose a different run number than 1
                        run_number = 5
                        predictions, labels = self.plot_helper_la_swa.get_predictions_and_labels_for_percentage(
                            percentage=run, idx=run_number, path_name_criteria=criteria,
                            laplace=True if 'laplace' in criteria else False
                        )
                    else:
                        predictions, labels = self.plot_helper_la_swa.get_predictions_and_labels_for_percentage(
                            percentage=run, idx=idx, path_name_criteria=criteria,
                            laplace=True if 'laplace' in criteria else False
                        )
                    if idx == 0:
                        preds = copy.deepcopy(predictions)
                        labs = copy.deepcopy(labels)
                    else:
                        preds = np.vstack((preds, predictions))
                        labs = np.concatenate((labs, labels))
                plot_calibration(preds, labs, ax=ax, label=f'{run} pct')
            set_legends_to_plot(ax)
            ax.set_title(f'Average Calibration {method}, {data_name}')
            # fig.tight_layout()
            if save_path is not None:
                fig.savefig(os.path.join(save_path, f'{criteria}_calibration_{data_name}_num_runs_{num_runs}.pdf'),
                            format='pdf', bbox_inches='tight', pad_inches=0.05)

            self.show()

    def plot_calibration_nodes(self, percentages=None, save_path = None):

        if percentages is None:
            percentages = ['1', '5', '61', '100']
        save_path = save_path if save_path is not None else self.save_path
        data_name = self.find_data_name(self.vi_hmc_path)
        num_runs = 15
        for method, criteria in [('Additive', 'add'), ('Multi.', 'node_run')]:
            fig, ax = plt.subplots(1, 1)
            for run in percentages:
                for idx in range(num_runs):
                    if num_runs == 1:
                        # Choose a different run number than 1
                        run_number = 5
                        predictions, labels = self.plot_helper_vi_hmc.get_predictions_and_labels_for_percentage(
                            percentage=run, idx=run_number, path_name_criteria=criteria,
                            laplace=True if 'laplace' in criteria else False
                        )
                    else:
                        predictions, labels = self.plot_helper_vi_hmc.get_predictions_and_labels_for_percentage(
                            percentage=run, idx=idx, path_name_criteria=criteria,
                            laplace=True if 'laplace' in criteria else False
                        )
                    if idx == 0:
                        preds = copy.deepcopy(predictions)
                        labs = copy.deepcopy(labels)
                    else:
                        preds = np.vstack((preds, predictions))
                        labs = np.concatenate((labs, labels))
                plot_calibration(preds, labs, ax=ax, label=f'{run} pct')
            set_legends_to_plot(ax)
            ax.set_title(f'Average Calibration {method}, {data_name}')
            # fig.tight_layout()
            if save_path is not None:
                fig.savefig(os.path.join(save_path, f'{criteria}_calibration_{data_name}_num_runs_{num_runs}.pdf'),
                            format='pdf', bbox_inches='tight', pad_inches=0.05)

            self.show()
    @staticmethod
    def find_best_and_make_bold(df, column, use_min = False):

        values = list(df[column])
        if use_min:
            best = np.argmin(values)
        else:
            best = np.argmax(values)

        new_values = []
        for idx, val in enumerate(values):
            if idx == best:
                format_ = f"\\textbf{'{' + '{:.3f}'.format(val) + '}'}"
            else:
                format_ = '{:.3f}'.format(val)
            new_values.append(format_)
        df[column] = new_values
        return df

    def plot_correlation_matrix(self):
        metrics_la = np.array(self.plot_helper_la_swa.run_for_dataset(criteria='laplace', laplace=True))[:, 1:]
        metrics_swa = np.array(self.plot_helper_la_swa.run_for_dataset(criteria='swag', laplace=False))[:, 1:]
        metrics_vi = np.array(self.plot_helper_vi_hmc.run_for_dataset(criteria='vi_run'))[:, 1:]
        metrics_hmc = np.array(self.plot_helper_vi_hmc.run_for_dataset(criteria='hmc'))[:, 1:]
        metrics_add = np.array(self.plot_helper_vi_hmc.run_for_dataset(criteria='add'))[:, 1:]
        metrics_mul = np.array(self.plot_helper_vi_hmc.run_for_dataset(criteria='node_run'))[:, 1:]

        runs = np.zeros_like(metrics_mul)
        runs = runs + np.array(range(metrics_mul.shape[0])).reshape(-1, 1)
        percentiles = np.zeros_like(runs) + np.array([1,2, 5, 8, 14, 23, 37, 61, 100]).reshape(1, -1)

        dataframe = pd.DataFrame()
        dataframe['runs'] = runs.ravel()
        dataframe['runs'] = dataframe['runs'].astype('category')
        dataframe['Perc.'] = percentiles.ravel()

        methods = ['LA', 'SWAG', 'VI', 'HMC', 'Add.', 'Mult.']

        fig, ax = plt.subplots(1,1)

        for method, metric in zip(methods,
                                  [metrics_la, metrics_swa, metrics_vi, metrics_hmc, metrics_add, metrics_mul]):

            dataframe[method] = -metric.ravel()

        cormat = dataframe.corr()
        # sns.color_palette("tab10", reverse=True, as_cmap=True)
        sns.heatmap(cormat, cmap="tab10_r",ax=ax, annot=True, fmt='.3f', annot_kws={"size": 14},
                    linewidths=.5, vmin=-1, vmax=1, cbar=False)
        data_name = self.find_data_name(self.la_swa_path)

        ax.set_title(f'Correlation Matrix, {data_name}')
        # ax.xticks(rotation=45)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=90, ha='right')
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        # plt.yticks(rotation=90)
        ax.tick_params(axis='y', labelrotation=1)

        if self.save_path is not None:
            fig.savefig(os.path.join(self.save_path, f'correlationmatrix_{data_name}.pdf'),
                        format='pdf')
        plt.show()


    def get_calibration_results(self, method, criteria):

        percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
        percentages = [str(p) for p in percentages]
        fig, ax = plt.subplots(1, 1)
        results = np.zeros((9, ))
        for i, perc in enumerate(percentages):
            predictions, labels = [], []
            for idx in range(15):
                if 'laplace' in criteria or 'swag' in criteria:
                    predictions_, labels_ = self.plot_helper_la_swa.get_predictions_and_labels_for_percentage(
                        percentage=perc, idx=idx, path_name_criteria=criteria,
                        laplace=True if 'laplace' in criteria else False
                    )
                else:
                    predictions_, labels_ = self.plot_helper_vi_hmc.get_predictions_and_labels_for_percentage(
                        percentage=perc, idx=idx, path_name_criteria=criteria,
                        laplace=True if 'laplace' in criteria else False
                    )
                predictions.append(predictions_)
                labels.append(labels_)

            predictions = np.concatenate(predictions, 0)
            labels = np.concatenate(labels, 0)
            fmu, fstd = predictions.mean(-1), predictions.std(-1)
            results[i] = uct.metrics.miscalibration_area(fmu, fstd, labels)
            # uct.plot_calibration(fmu, fstd, labels, ax=ax)
        return results


    def write_calibration_latex_table(self, estimator = np.median, save_path = None, bold_direction = 'percentage'):

        metrics_vi = self.get_calibration_results('VI', 'vi_run')
        metrics_la = self.get_calibration_results('Laplace', 'laplace')
        metrics_swa = self.get_calibration_results('SWAG', 'swag')
        metrics_hmc = self.get_calibration_results('HMC', 'hmc')
        metrics_add = self.get_calibration_results('Additive', 'add')
        metrics_mul = self.get_calibration_results('Multi.', 'node_run')

        percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
        methods = ['Laplace', 'SWAG', 'VI', 'HMC', 'Additive', 'Multi.']
        use_min = True
        df = pd.DataFrame()
        for method, metric in zip(methods,
                                  [metrics_la, metrics_swa, metrics_vi, metrics_hmc, metrics_add, metrics_mul]):
            df[method] = metric

        if bold_direction == 'method':
            for column in df.columns:
                df = self.find_best_and_make_bold(df, column, use_min)
        df = df.transpose()
        df = df.rename(columns={i: f"{perc}." for i, perc in zip(df.columns, percentages)}).round(decimals=2)
        if bold_direction == 'percentage':
            for column in df.columns:
                df = self.find_best_and_make_bold(df, column, use_min)
        print(r"\begin{table}", "\n", '\centering \n', '\caption{} \n',
              df.to_latex(index=True, float_format="{{:0.2f}}".format, escape=False),
              '\label{} \n', r"\end{table}", "\n")


    def write_latex_table(self,estimator = np.median,  save_path = None, bold_direction = 'percentage', ndigits=2):
        metrics_vi = np.array(self.plot_helper_vi_hmc.run_for_dataset(criteria='vi_run'))[:, 1:]
        metrics_la = np.array(self.plot_helper_la_swa.run_for_dataset(criteria='laplace', laplace=True))[:, 1:]
        metrics_swa = np.array(self.plot_helper_la_swa.run_for_dataset(criteria='swag', laplace=False))[:, 1:]
        metrics_hmc = np.array(self.plot_helper_vi_hmc.run_for_dataset(criteria='hmc'))[:, 1:]
        metrics_add = np.array(self.plot_helper_vi_hmc.run_for_dataset(criteria='add'))[:,1:]
        metrics_mul = np.array(self.plot_helper_vi_hmc.run_for_dataset(criteria='node_run'))[:, 1:]

        metrics_la = estimator(metrics_la, 0)
        metrics_swa = estimator(metrics_swa, 0)
        metrics_hmc = estimator(metrics_hmc, 0)
        metrics_vi = estimator(metrics_vi, 0)
        metrics_add = estimator(metrics_add, 0)
        metrics_mul = estimator(metrics_mul, 0)
        percentages = self.percentages[1:]

        use_min = True

        methods = ['Laplace', 'SWAG', 'VI', 'HMC', 'Additive', 'Multiplicative']
        df = pd.DataFrame()
        for method, metric in zip(methods, [metrics_la, metrics_swa, metrics_vi, metrics_hmc, metrics_add, metrics_mul]):
            if self.plot_helper_vi_hmc.eval_method in ['nll', 'nll_glm', 'glm_nll']:
                df[method] = -metric
                continue
            df[method] = metric

        if bold_direction == 'method':
            for column in df.columns:
                df = self.find_best_and_make_bold(df, column, use_min)
        df = df.transpose()
        df = df.rename(columns = {i: f"{perc}." for i, perc in zip(df.columns, percentages)})
        if bold_direction == 'percentage':
            for column in df.columns:
                df = self.find_best_and_make_bold(df, column, use_min)

        print(r"\begin{table}", "\n", '\centering \n', '\caption{} \n',
              df.to_latex(index=True, float_format="{:.3f}".format, escape=False),
              '\label{} \n', r"\end{table}", "\n")


class ExtendedString(str):
    pass


if __name__ == '__main__':
    # path_la = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_Laplace_MAP'
    # path_swag = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI_SWAG_MAP_nobayes'
    # plot_la_swag(path_la, path_swag)

    # # PETER PATHS
    # path_la = r'C:\Users\45292\Documents\Master\UCI_Laplace_SWAG_all_metrics\UCI_Laplace_SWAG_all_metrics\energy_models'
    #
    # path_vi =r'C:\Users\45292\Documents\Master\HMC_VI_TORCH_FIN\UCI_HMC_VI_torch\energy_models'
    #
    # plot_holder = PlotFunctionHolder(path_la, path_vi)
    # plot_holder.write_calibration_latex_table()
    # breakpoint()
    # plot_holder = PlotFunctionHolder(path_la, path_vi, eval_method='nll', calculate=True,
    #                                  save_path=r'C:\Users\45292\Documents\Master\Figures\UCI\HMC')
    # plot_holder.plot_prior_laplace()
    # breakpoint()
    # # plot_holder.write_latex_table()
    # # plot_holder.plot_number_of_parameters(save_path=r'C:\Users\45292\Documents\Master\Figures\UCI')
    # save_path = r'C:\Users\45292\Documents\Master\Figures\UCI\HMC'
    # plot_holder.plot_hmc_sample_scaling(save_path=save_path)
    # plot_holder.set_eval_method('mse')
    # plot_holder.plot_hmc_sample_scaling(save_path=save_path)
    #
    # # plot_holder.plot_partial_percentages_vi_hmc()
    # breakpoint()
    # plot_holder.plot_partial_percentages_la_swa(save_path=r'C:\Users\45292\Documents\Master\Figures\UCI\Laplace')
    # breakpoint()
    # # plot_holder = PlotFunctionHolder(la_swa_path=path_la, vi_hmc_path=path_vi, calculate=True)
    # # plot_holder.plot_partial_percentages_la_swa()

    # APPLY TO LA-SWAG PATH ONCE
    # change_datasets(path_la)

    # GUSTAV PATHS
    path_la = r'C:\Users\45292\Documents\Master\UCI_Laplace_SWAG\KFAC\boston'
    save_path = r'C:\Users\45292\Documents\Master\UCI_Laplace_SWAG\KFAC\Figures\Boston'
    plot_holder = PlotFunctionHolder(la_swa_path=path_la, calculate=False,
                                     save_path=save_path,
                                     eval_method='nll_glm')
    plot_holder.plot_partial_percentages_kron()

    breakpoint()
    path_la = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI\UCI_Laplace_SWAG_all_metrics'
    path_vi = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI\UCI_HMC_VI_torch'

    path_la_var = r"C:\Users\Gustav\Desktop\MasterThesisResults\UCI\UCI_Laplace_SWAG_all_metrics_var_mask"

    path_la_rand = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI\UCI_Laplace_SWAG_all_metrics_rand'
    path_vi_rand = r'C:\Users\Gustav\Desktop\MasterThesisResults\UCI\UCI_HMC_VI_torch_rand'

    # datasets = ['yacht']
    metric = 'nll'
    datasets = ['boston', 'energy', 'yacht']
    prediction_folders = [ dataset + "_models" for dataset in datasets]

    # NORMAL PLOTTING, WITH CORRECT NLL CALCULATION
    for prediction_folder in prediction_folders:
        la_swa_path = os.path.join(path_la, prediction_folder)
        vi_hmc_path = os.path.join(path_vi, prediction_folder)

        la_swa_path_rand = os.path.join(path_la_rand, prediction_folder)
        vi_hmc_path_rand = os.path.join(path_vi_rand, prediction_folder)

        la_var_path = os.path.join(path_la_var, prediction_folder)

        save_path = os.path.join(os.getcwd(), r"Figures\Calibration")
        plot_holder = PlotFunctionHolder(la_swa_path=la_swa_path, vi_hmc_path=vi_hmc_path, calculate=True, save_path=save_path,
                                         la_swa_path_rand=la_swa_path_rand, vi_hmc_path_rand=vi_hmc_path_rand, eval_method=metric,
                                         la_var_path=la_var_path)

        # plot_holder.plot_partial_percentages_la_vs_la_var_select()
        # plot_holder.plot_pred_labels_vi_hmc()
        # plot_holder.plot_pred_labels_la_swa()
        # plot_holder.plot_pred_labels_node_based()
        #
        # plot_holder.plot_calibration_vi_hmc()
        # plot_holder.plot_calibration_la_swa()
        # plot_holder.plot_calibration_nodes()

        plot_holder.plot_correlation_matrix()
        # plot_holder.write_calibration_latex_table()
        # plot_holder.write_latex_table()

        # plot_holder.plot_partial_percentages_nodes()
        # plot_holder.plot_partial_percentages_vi_hmc()
        # plot_holder.plot_partial_percentages_rand_vs_max(criteria='node_run')
        # plot_holder.plot_partial_percentages_rand_vs_max(criteria='add')

        # plot_holder.plot_partial_percentages_rand_vs_max(criteria='node_run', map=False)
        # plot_holder.plot_partial_percentages_rand_vs_max(criteria='add', map=False)
        # plot_holder.plot_partial_percentages_rand_vs_max(criteria='laplace', map=False)
        # plot_holder.plot_partial_percentages_rand_vs_max(criteria='hmc', map=False)
        # plot_holder.plot_partial_percentages_rand_vs_max(criteria='swag', map=False)
        # plot_holder.plot_partial_percentages_rand_vs_max(criteria='vi_run', map=False)

        # if 'yacht' in prediction_folder:
        #     plot_holder.plot_partial_percentages_la_vs_la_var_select(map=False)
            # plot_holder.plot_partial_percentages_nodes(map=False)
            # plot_holder.plot_partial_percentages_la_swa(map=False)

    breakpoint()
    #
    # breakpoint()