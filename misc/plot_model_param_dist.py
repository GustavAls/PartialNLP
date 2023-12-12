import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import os
import torch.nn as nn
import torch
import matplotlib as mpl
from MAP_baseline.MapNN import MapNN

def read_model(path):
    model_dict = pickle.load(open(path, 'rb'))
    return model_dict

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

def plot_layer(layer, ax = None, percentile_values = None, max_value = None):

    if ax is None:
        fig, ax = plt.subplots(1,1)

    all_percentiles = [1, 2, 5, 8, 14, 23, 37, 61, 100]
    heatmap = np.zeros_like(layer)
    for idx, value in enumerate(percentile_values):
        heatmap[(value < layer) & (layer < max_value)] = all_percentiles[idx]
        max_value = value


    if heatmap.shape[0] == 1:
        heatmap = heatmap.T
    sns.heatmap(data=heatmap, ax = ax, square=True)
    # plt.show()


def plot_only_percentile(layer):
    all_percentiles = [1, 2, 5, 8, 14, 23, 37, 61, 100]

    maximum = np.max(layer)
    heatmap = np.zeros_like(layer)
    for idx, perc in enumerate(all_percentiles):
        minimum = np.percentile(layer, q=100 - perc)
        heatmap[(minimum < layer) & (layer < maximum)] = 100 - perc
        maximum = minimum

        sns.heatmap(data=heatmap, vmax=100, vmin=0, square=True)
        plt.show(block = False)


def read_model_(path, dataset = 'Energy'):

    if dataset == 'Energy':
        mle_model = MapNN(input_size=8, width=50, output_size=1, non_linearity="leaky_relu")
    elif dataset == 'Boston':
        mle_model = MapNN(input_size=13, width=50, output_size=1, non_linearity="leaky_relu")
    else:
        mle_model = MapNN(input_size=6, width=50, output_size=1, non_linearity="leaky_relu")

    mle_model.load_state_dict(torch.load(path))

    linears = {}
    for name, module in mle_model.named_modules():
        if isinstance(module, nn.Linear):
            linears[name] = module.weight.data.numpy()

    return linears


def get_linears_for_all_models(dataset = 'Energy'):
    if dataset == 'Energy':
        map_path = r'C:\Users\45292\Documents\Master\UCI_Laplace_SWAG\energy_data\energy'
    elif dataset == 'Yacht':
        map_path = r'C:\Users\45292\Documents\Master\ModelsForHeatmapPlotting\yacht'
    else:
        map_path = r'C:\Users\45292\Documents\Master\ModelsForHeatmapPlotting\boston'

    files = [os.path.join(map_path, p) for p in os.listdir(map_path)]
    linears = []
    for file in files:
        linears.append(read_model_(file, dataset=dataset))

    return linears

def read_all_models_and_run(all_datasets = False,  dataset = None):


    if all_datasets:
        df_all = []
        for dataset in ['Energy', 'Boston', 'Yacht']:
            linears = get_linears_for_all_models(dataset=dataset)
            df = calculate_and_concatenate_into_df(linears)
            df_all.append(df)
            df_all = pd.concat(df_all)
            plot_df(df_all, name='All')
    elif dataset is not None:
        linears = get_linears_for_all_models(dataset=dataset)
        df = calculate_and_concatenate_into_df(linears)
        plot_df(df, name = dataset)




def get_percentiles_for_model(model_dict):
    percentiles = [1, 2, 5, 8, 14, 23, 37, 61, 100]
    values = np.concatenate([val.ravel() for val in model_dict.values()])
    results_dict = {k: [] for k in model_dict.keys()}
    for k, v in model_dict.items():
        for p in percentiles:
            percentage_value = np.percentile(values, 100 - p)
            results_dict[k].append(np.sum(v >= percentage_value)/np.prod(v.shape)*100)
    results_dict['percentages'] = percentiles
    return results_dict


def plot_df(df, name = 'all'):

    layer_one_col = 'tab:red'
    layer_two_col = 'tab:green'
    layer_three_col = 'tab:blue'

    fig, ax = plt.subplots(1,1)
    errorbar_func = lambda x: np.percentile(x, [25, 75])
    sns.pointplot(errorbar=errorbar_func,
                  data=df, x="percentages", y='layer one',
                  join=False,
                  capsize=.30,
                  markers="d",
                  scale=1.0,
                  err_kws={'linewidth': 0.7}, estimator=np.median,
                  color=layer_one_col,
                  label='Layer one',
                  ax=ax)
    sns.pointplot(errorbar=errorbar_func,
                  data=df, x="percentages", y='layer two',
                  join=False,
                  capsize=.30,
                  markers="d",
                  scale=1.0,
                  err_kws={'linewidth': 0.7}, estimator=np.median,
                  color=layer_two_col,
                  label='Layer two',
                  ax=ax)
    sns.pointplot(errorbar=errorbar_func,
                  data=df, x="percentages", y='layer three',
                  join=False,
                  capsize=.30,
                  markers="d",
                  scale=1.0,
                  err_kws={'linewidth': 0.7}, estimator=np.median,
                  color=layer_three_col,
                  label='Classifier',
                  ax=ax)

    df_other = pd.DataFrame()
    df_other['percentages'] = df['percentages']
    df_other['y'] = df['percentages']
    sns.pointplot(errorbar=None,
                  data=df_other, x="percentages", y='y',
                  join=True,
                  capsize=.30,
                  scale=0.5,
                  linewidth=0.7,
                  err_kws={'linewidth': 0.5}, estimator=np.median,
                  color="tab:orange",
                  label='Uniform',
                  ax=ax)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
              ncol=2, fancybox=True, shadow=True)
    ax.set_ylabel('Percentage of stoch. params')
    ax.set_title(f'Layer-wise percentage of stoch. params. - {name}')
    # ax.set_aspect('equal', adjustable='datalim')
    # ax.set_yscale('log')
    # ax.set_yticks(df['percentages'])
    fig.tight_layout()
    fig.savefig(fr'C:\Users\45292\Documents\Master\ModelsForHeatmapPlotting\{name}_stoch_layer.pdf', format='pdf')
    plt.show()
def calculate_and_concatenate_into_df(linears):

    keys = list(linears[0].keys())
    result_holder = {k: [] for k in keys}
    result_holder['run'] = []
    result_holder['percentages'] = []
    for idx, linear in enumerate(linears):
        res_dict = get_percentiles_for_model(linear)
        for k, v in res_dict.items():
            result_holder[k] += v
        result_holder['run'] += [idx]*len(v)

    df = pd.DataFrame()
    df['layer one'] = result_holder[keys[0]]
    df['layer two'] = result_holder[keys[1]]
    df['layer three'] = result_holder[keys[2]]
    df['runs'] = result_holder['run']
    df['percentages'] = result_holder['percentages']

    return df



def read_an_plot(path):

    model_dict = read_model(path)
    values = np.concatenate([val['linear'].ravel() for val in model_dict.values()])
    percentile_values = [np.percentile(values, q=100-perc) for perc in [1, 2, 5, 8, 14, 23, 37, 61, 100]]
    maximum = np.max(values)
    for module, submod in model_dict.items():
        fig, ax = plt.subplots(1,1)
        linear_layer = submod['linear'].numpy()
        plot_layer(linear_layer, ax = ax, percentile_values = percentile_values, max_value = maximum)
        # ax.set_title(module)
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        ax.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left = False,  # ticks along the bottom edge are off
            right = False,  # ticks along the top edge are off
            labelleft=False)
        plt.show(block = False)

if __name__ == '__main__':

    for dataset in ['Energy', 'Yacht', 'Boston']:
        read_all_models_and_run(dataset=dataset)
    breakpoint()
    read_an_plot(r"C:\Users\45292\Documents\Master\ModelsForHeatmapPlotting\model_energy_0.pkl")
    breakpoint()
    test = np.random.normal(0,1, (20, 20))


    plot_layer(test)
    breakpoint()
