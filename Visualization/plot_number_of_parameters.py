import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_number_of_parameters_normal(max_num,  percentile):
    return max_num * percentile / 100

def model_value_dist(matrices, all_values, percentile):
    p_val = np.percentile(np.abs(all_values), 100 - percentile)
    masks = [np.abs(mat) >= p_val for mat in matrices]
    return masks


def get_model_values(matrix_sizes):
    matrices = [np.random.normal(0, 1, size=size) for size in matrix_sizes]
    all_values = np.concatenate([mat.ravel() for mat in matrices])
    return matrices, all_values
def get_active_nodes(mask):

    s = np.zeros((mask.shape[0], ))
    r = np.zeros((mask.shape[1], ))

    for i in range(mask.shape[0]):
        if np.any(mask[i, :]):
            s[i] = 1
    for j in range(mask.shape[1]):
        if np.any(mask[:, j]):
            r[j] = 1

    return s, r


def simulate_n_times(num_sims = 100):

    percentiles = [1, 2, 5, 8, 14, 23, 37, 61, 100]
    num_parameters =[]
    matrix_sizes = [(8, 50), (50, 50), (50, 1)]
    results = np.zeros((len(percentiles), num_sims))
    for idx, p in enumerate(percentiles):
        for i in range(num_sims):
            matrices, all_values = get_model_values(matrix_sizes)

            masks = model_value_dist(matrices, all_values, p)
            counter = 0
            for mask in masks:
                s,r = get_active_nodes(mask)
                counter += s.sum() + r.sum()
            results[idx, i] = counter

    return results


def plot_with_error_bars(percentile_mat, path, show_big = True, ax = None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    percentiles = [1, 2, 5, 8, 14, 23, 37, 61, 100]
    with sns.axes_style("whitegrid"):
        ax.plot(percentiles, percentile_mat.mean(-1), color = 'tab:blue',
                label = 'Node Based Additive', linestyle = 'dashed')
        lower, upper = np.percentile(percentile_mat, (0, 100), axis=-1)
        ax.fill_between(percentiles,lower, upper, alpha = 0.5, color = 'tab:blue')
        ax.plot(percentiles, [209*perc/100 for perc in percentiles], color = 'tab:red',
                linestyle = 'dashed', label = 'Node-Based Multiplicative')
        if show_big:
            ax.plot(percentiles, [3000 * perc/100 for perc in percentiles], color = 'tab:orange',
                    linestyle = 'dashed', label = 'LSVH')
            ax.set_yscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel('Percentiles')
        ax.set_ylabel('Num. Stoch Trainable Params')
        ax.set_title('Num. trainable stochastic paramaters by method')
        ax.grid(linewidth = 1, alpha = 0.7)


if __name__ == '__main__':

    res = simulate_n_times(num_sims=100)
    plot_with_error_bars(percentile_mat=res, path = r"C:\Users\45292\Desktop\without_full.pdf")
    breakpoint()
    plt.hist(simulate_n_times(num_sims=10000), bins=20, density=True)
    plt.show()
    breakpoint()




