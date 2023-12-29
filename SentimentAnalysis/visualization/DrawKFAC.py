import os

import matplotlib.pyplot as plt
from scipy.stats import wishart, multivariate_normal
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
def make_big_matrix(num_modules, module_sizes, sparse = False, percentile = 0.0):

    if isinstance(module_sizes, int):
        module_sizes = [module_sizes for _ in range(num_modules)]

    covariance_matrices = []

    for i in range(num_modules):
        # wish_ = wishart(module_sizes[i], np.eye(module_sizes[i]) * 20)
        #
        # R = wish_.rvs()
        # s_diag = np.zeros(R.shape)
        # np.fill_diagonal(s_diag, R.diagonal())
        cov_mat_1 = np.random.uniform(0, 1, size = (module_sizes[i], module_sizes[i]))
        cov_mat_2 = np.random.uniform(0, 1, size = (module_sizes[i], module_sizes[i]))
        if sparse:
            if percentile == 0:
                percentile = 0.8

            rows = np.random.choice(range(module_sizes[i]), size = (int(module_sizes[i] * (1-percentile)),), replace=False)
            columns = np.random.choice(range(module_sizes[i]), size=(int(module_sizes[i]* (1-percentile)),), replace=False)
            cov_mat_1[rows, :] = 0
            cov_mat_1[:, rows] = 0
            cov_mat_1[columns, :] = 0
            cov_mat_1[:, columns] = 0
            cov_mat_full = np.kron(cov_mat_1, cov_mat_2)
            lower = np.tril_indices(cov_mat_full.shape[0], -1)
            cov_mat_full[lower] = cov_mat_full.T[lower]

        covariance_matrices.append(cov_mat_full)

    size = sum((covariance_matrices[i].shape[0] for i in range(len(covariance_matrices))))
    full_matrix = np.zeros((size, size))

    counter = [0, 0]
    for idx, cov_mat in enumerate(covariance_matrices):
        shape = cov_mat.shape
        full_matrix[counter[0]:counter[0] + shape[0], counter[1]: counter[1] + shape[1]] = cov_mat
        counter[0] += shape[0]
        counter[1] += shape[1]

    return full_matrix


def draw_full_matrix(full_matrix, save_path = None):

    # sns.heatmap(data=full_matrix, vmin=0,yticklabels=False, xticklabels=False, ax = ax,
    #             cbar=False,  cmap = sns.color_palette("light:b", as_cmap=True))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # full_matrix = np.abs(full_matrix-1)

    cmap0 = LinearSegmentedColormap.from_list('', ['black', 'white'])

    plt.imshow(full_matrix, cmap = cmap0)
    plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format = 'pdf')
    plt.show()


if __name__ == '__main__':
    save_path = r'C:\Users\45292\Documents\Master\SentimentClassification\IMDB Figures\Model General'
    for percentile in [0.4, 0.5, 0.8, 0.9]:
        big_matrix = make_big_matrix(2, [100, 100], sparse=True, percentile = percentile)
        draw_full_matrix(big_matrix, os.path.join(save_path, f'kfac_cov_perc_{percentile}.pdf'))
