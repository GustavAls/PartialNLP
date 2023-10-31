import ast
import sys, os, time, requests

sys.path.append(os.getcwd())
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, Trace_ELBO, autoguide, SVI
from uci import UCIDataloader
from torch.utils.data import DataLoader
from VI.partial_bnn_functional import train
from MAP_baseline.MapNN import MapNN
from misc.likelihood_losses import GLLGP_loss_swag, BaseMAPLossSwag
from torch.nn import MSELoss
import pickle
import jax
import jax.numpy as jnp
import jax.nn
from Laplace.uci_laplace import calculate_std
from jax import random

numpyro.set_platform("cpu")
numpyro.set_host_device_count(8)
import argparse
from torch.distributions import Normal
from HMC.uci_hmc import UCIDataset, UCIBostonDataset, UCIEnergyDataset, UCIYachtDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

class PlotHelper:

    def __init__(self, path_to_models, eval_method='nll', calculate = True):
        self.path_to_models = path_to_models
        self.pca = None
        self.scaler = None

        self.eval_method = eval_method
        self.calculate = calculate
    def inverse_transform(self, x):
        if self.scaler is None:
            return x
        else:
            return self.scaler.inverse_transform(x)

    def get_predictions_and_labels_for_percentage(self, percentage, idx = 0, path_name_criteria = ""):
        paths = self.get_paths(self.path_to_models, criteria=path_name_criteria)
        path = paths[idx]

        pcl = pickle.load(open(path, 'rb'))
        labels = pcl['dataset'].y_test.flatten()
        train, val, test =self.convert_to_proper_format(pcl[percentage])

        return test, labels

    def plot_scatter(self, predictions, labels):
        pass
    def PCA(self, X, scaler=None, x_test=None):

        if scaler is not None:
            self.scaler = scaler
            x_trans = scaler.inverse_transform(X)
        else:
            x_trans = X.copy()
        self.pca = PCA(n_components=2).fit(x_trans)
        if x_test is not None:
            return self.pca.transform(x_test)
        return None

    def plot(self, fmu, fvar, labels, scale, loc, x_test=None):
        x_axis = range(len(labels))
        fmu, fvar, labels = fmu.flatten(), fvar.flatten(), labels.flatten()
        if x_test is not None and x_test.ndim > 1:
            if self.pca is not None:
                x_axis = self.pca.transform(self.inverse_transform(x_test))[:, 0]
            else:
                UserWarning("x_test provided to the function without PCA having been fitted")

        fig, ax = plt.subplots(1, 1)
        ax.plot(x_axis, labels, 'bo', label='Labels')
        ax.plot(x_axis, fmu * scale + loc, 'k*', label='Predictions')
        ax.fill_between(x_axis, fmu * scale + loc - fvar * scale, fmu * scale + loc + fvar * scale, alpha=0.25)
        ax.set_ylim(*np.percentile(np.concatenate((fmu * scale + loc - fvar * scale, fmu * scale + loc + fvar * scale,
                                                   labels)), q=[0, 100]))
        ax.legend()

        plt.show()

    def glm_predictive(self, preds, std=True):
        fmu = preds.mean(1)
        if std:
            f_var = preds.std(1)
        else:
            f_var = preds.var(1)

        return fmu, f_var

    def glm_nll(self, fmu, fvar, labels, scale, loc):
        nlls = []
        labels = torch.from_numpy(labels * scale + loc)
        for mu, var, lab in zip(fmu, fvar, labels):
            dist = Normal(mu * scale + loc, var * scale)
            nlls.append(
                dist.log_prob(lab.view(1, 1)).item()
            )

        return np.mean(nlls)

    def get_data(self):
        nlls = np.array(self.run_for_dataset())
        percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
        if np.min(nlls.shape) == 10:
            percentages = [0] + percentages
        return nlls, percentages

    def get_residuals(self, predictions, labels, full=False):
        if full:
            return predictions - labels
        return predictions.mean(1).flatten() - labels.flatten()

    def calculate_nll_(self, mc_matrix, labels, y_scale, y_loc, sigma):
        results = []
        labs = torch.from_numpy(labels * y_scale + y_loc)

        for i in range(mc_matrix.shape[0]):
            res_temp = []
            for j in range(mc_matrix.shape[1]):
                dist = Normal(mc_matrix[i, j] * y_scale + y_loc, np.sqrt(sigma) * y_scale)
                res_temp.append(dist.log_prob(labs[i].view(1, 1)).item())
            results.append(np.mean(res_temp))
        return np.mean(results)

    @staticmethod
    def convert_to_proper_format(run):

        preds_train = np.asarray(run['predictive_train']).squeeze(-1).transpose(1, 0)
        preds_val = np.asarray(run['predictive_val']).squeeze(-1).transpose(1, 0)
        preds_test = np.asarray(run['predictive_test']).squeeze(-1).transpose(1, 0)

        return preds_train, preds_val, preds_test

    @staticmethod
    def get_labels(dataset):
        y_train, y_val, y_test = dataset.y_train, dataset.y_val, dataset.y_test
        return y_train, y_val, y_test

    @staticmethod
    def get_sigma(residuals):
        return residuals.std()

    def run_for_key(self, pcl, key):

        dataset = pcl['dataset']
        y_train, y_val, y_test = self.get_labels(dataset)
        preds_train, preds_val, preds_test = self.convert_to_proper_format(pcl[key])
        sigma = self.get_sigma(self.get_residuals(preds_train, y_train, full=True))
        mse = np.mean(self.get_residuals(preds_test, y_test) ** 2)

        if not self.calculate and self.eval_method != 'all':
            return pcl[key][self.eval_method]

        if self.eval_method == 'all':
            metric = {}
            for eval_method in ['nll', 'elpd', 'elpd_sqrt']:
                self.eval_method = eval_method
                metric[eval_method] = self.run_for_key(pcl, key)
            self.eval_method = 'all'


        if self.eval_method in  ['nll', 'nll_glm', 'glm_nll']:

            fmu, fvar = self.glm_predictive(preds_test, std=False)

            if 'map' in key:
                fvar = np.zeros_like(fmu)
            metric = self.glm_nll(fmu, fvar + sigma, y_test,
                                  dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item())

        elif self.eval_method == 'elpd':
            metric = self.calculate_nll_(
                preds_test, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(), sigma ** 2
            )
        elif self.eval_method == 'elpd_sqrt':
            metric = self.calculate_nll_(
                preds_test, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(), sigma ** 2
            )

        elif self.eval_method == 'mse':
            metric = self.calculate_mse(
                preds_test, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item()
            )

        if 'map' in key and self.eval_method in ['elpd']:
            metric = self.calculate_nll_(preds_test, y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(),
                                         sigma ** 2)
        return metric

    def run_for_all_keys(self, pcl):
        all_keys = ['map_results', '1', '2', '5', '8', '14', '23', '37', '61', '100']
        nlls = []
        for key in all_keys:
            nlls.append(self.run_for_key(pcl, key))
        return nlls

    def calculate_mse(self, preds, labels, scale, loc):
        pred = preds
        label = labels
        return np.square(pred - label).mean()
    def run_for_multiple_files(self, paths):

        nlls = []
        pbar = tqdm(paths, desc='Running through files')
        for path in pbar:
            pcl = pickle.load(open(path, 'rb'))
            nlls.append(self.run_for_all_keys(pcl))
        return nlls

    def get_paths(self, path, criteria=None):
        if criteria is None:
            criteria = ""
        paths = [os.path.join(path, p) for p in os.listdir(path) if criteria in p]
        return paths

    def run_for_dataset(self, criteria=None):

        paths = self.get_paths(self.path_to_models, criteria)
        nlls = self.run_for_multiple_files(paths)
        return nlls
