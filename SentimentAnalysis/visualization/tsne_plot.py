import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.datasets import make_s_curve
from scipy.stats import wishart, multivariate_normal
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from sklearn.linear_model import LogisticRegression
import seaborn as sns
def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


class T_SNEPlotter:
    def __init__(self, embeddings = None, labels = None, manifold = 'pca'):

        if embeddings is None and labels is None:
            self.embeddings, self.labels, self.embeddings_test, self.labels_test = self.make_simulation()

        if manifold == 'tsne':
            t_sne = TSNE(
                n_components=2,
                perplexity=30,
                init="random",
                n_iter=250,
                random_state=0,
            )
            self.downsampled = t_sne.fit_transform(self.embeddings)
        elif manifold == 'pca':
            pca = PCA(n_components=2).fit(self.embeddings_test)
            self.downsampled = pca.transform(self.embeddings_test)
            self.log_reg = LogisticRegression().fit(self.embeddings, self.labels)
            self.predictions = self.log_reg.predict_proba(self.embeddings_test)


    def plot(self):
        plot_2d(self.downsampled, self.labels, "T-distributed Stochastic  \n Neighbor Embedding")

    def plot_grid(self):
        down = self.downsampled.copy()

        down = down + down.min(0)
        down = down / down.max(0)
        max_x, max_y = down.max(0)
        min_x, min_y = down.min(0)

        grid_x, grid_y = np.mgrid[min_x:max_x:100j, min_y:max_y:200j]
        fig, ax = plt.subplots(1,1)

        grid_z2 = griddata(down, self.predictions[:,0], (grid_x, grid_y), method='cubic', fill_value=0)
        grid_z2_label_1 = griddata(down, self.predictions[:, 1], (grid_x, grid_y), method='cubic', fill_value=0)
        grid_z2_label_1 /= np.max(grid_z2_label_1)
        grid_z2 /= np.max(grid_z2)
        showing = (grid_z2 - grid_z2_label_1) + 1
        ax.imshow(showing, vmin=showing.min(),
                      vmax=showing.max(), cmap='seismic')

        # ax.imshow(grid_z2_label_1, vmin=0, vmax=grid_z2_label_1.max(), cmap = 'Blues')
        fig.show()
        breakpoint()
        # plt.imshow(grid_z2, extent=(0, 1, 0, 1), origin='lower')

        plt.show()
        plt.imshow(grid_z2,extent=(0, 1, 0, 1), origin='lower')
        plt.show()
    def make_simulation(self):
        mean_1 = np.random.normal(0,1, (50, ))
        mean_2 = np.random.normal(0,1, (50, ))

        wish_ = wishart(50, np.eye(50)*0.1)
        gauss_one = multivariate_normal(mean_1, wish_.rvs())
        gauss_two = multivariate_normal(mean_2, wish_.rvs())

        embeddings = np.concatenate([gauss_one.rvs(500), gauss_two.rvs(500)], 0)
        embeddings_test = np.concatenate([gauss_one.rvs(500), gauss_two.rvs(500)],0)
        labels = np.concatenate([np.zeros(500), np.ones(500)], 0)
        labels_test = np.concatenate([np.zeros(500), np.zeros(500)], 0)
        return embeddings, labels, embeddings_test, labels_test



class MommyPlotter:

    def __init__(self):
        self.num_people = [10, 15, 20, 25]
        self.underlying_truths = np.linspace(1e-10, 1, 100)
        self.underlying_truths_other = [0.6, 0.7, 0.8, 0.9][2:]
    def simulate(self, num):
        results = []
        for truth in self.underlying_truths:
            simulation = np.random.binomial(1, p = truth, size = (num, int(1e5)))
            simulation = np.mean(simulation, 0) > 0.75
            results.append(np.sum(simulation)/len(simulation))
        return results

    def simulate_other(self, underlying_truth):
        results = []
        for num in range(5, 36):
            subsim_res = []
            print(num)
            simulation = np.random.binomial(1, p = underlying_truth, size = (num, int(1e3), int(1e4)))
            simulation = np.mean(np.mean(simulation, 0) > 0.5, 0) > 0.95
            results.append(np.mean(simulation))
        return results

    def get_other_for_all(self):
        colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
        fig, ax = plt.subplots(1, 1)
        df = pd.DataFrame()
        for num in self.underlying_truths_other:
            df[num] = self.simulate_other(num)

        df['sample size'] = np.array(range(5, 36))
        for idx, color in zip(self.underlying_truths_other, colors):
            ax.plot(df['sample size'], df[idx], color = color, label = f'Underlying truth {idx}')

        ax.legend()
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Probabi')
        ax.set_title('Probability of estimating P(CR > 0.75)')
        plt.show()





    def get_for_all(self):
        colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
        fig, ax = plt.subplots(1,1)
        df = pd.DataFrame()
        for num in self.num_people:
            df[str(num)] = self.simulate(num)
        df['underlying truth'] = self.underlying_truths
        for idx, color in zip(self.num_people, colors):
            ax.plot(self.underlying_truths, df[str(idx)], color = color, label = f'sample size {idx}')

        ax.legend()
        ax.set_xlabel('True underlying probabilty')
        ax.set_ylabel('Probability of estimating P(CR > 0.75)')
        ax.set_title('Probability of estimating P(CR > 0.75)')
        plt.show()
        fig, ax = plt.subplots(1,1)
        for idx, color in zip(self.num_people, colors):
            ax.plot(self.underlying_truths[:50], df[str(idx)][:50], color = color, label = f'sample size {idx}')

        ax.legend()
        ax.set_xlabel('True underlying probabilty')
        ax.set_ylabel('Probability of estimating P(CR > 0.75)')
        ax.set_title('Probability of estimating P(CR > 0.75) zoomed on lower')
        plt.show()

        fig, ax = plt.subplots(1,1)
        for idx, color in zip(self.num_people, colors):
            ax.plot(self.underlying_truths[-50:], df[str(idx)][-50:], color = color, label = f'sample size {idx}')

        ax.legend()
        ax.set_xlabel('True underlying probabilty')
        ax.set_ylabel('Probability of estimating P(CR > 0.75)')
        ax.set_title('Probability of estimating P(CR > 0.75) zoomed on higher')
        plt.show()



if __name__ == '__main__':

    plotter = T_SNEPlotter()
    plotter.plot_grid()
