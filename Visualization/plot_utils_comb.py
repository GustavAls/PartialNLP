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

class PlotHelper:

    def __init__(self, path_to_models):
        self.path_to_models = path_to_models


    def get_data(self):
        nlls = np.array(self.run_for_dataset())
        percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
        if np.min(nlls.shape) == 10:
            percentages = [0] + percentages
        return nlls, percentages

    def get_residuals(self, predictions, labels):
        return predictions.mean(1).flatten() - labels.flatten()

    def calculate_nll_(self, mc_matrix, labels, y_scale, y_loc, sigma):
        results = []
        labs = torch.from_numpy(labels * y_scale + y_loc)

        for i in range(mc_matrix.shape[0]):
            res_temp = []
            for j in range(mc_matrix.shape[1]):
                dist = Normal(mc_matrix[i, j] * y_scale + y_loc, np.sqrt(sigma) * y_scale)
                res_temp.append(dist.log_prob(labs[i].view(1,1)).item())
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
        sigma = self.get_sigma(self.get_residuals(preds_train, y_train))
        mse = np.mean(self.get_residuals(preds_test, y_test)**2)
        if 'map' in key:
            nll = self.calculate_nll_(preds_test, y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(), sigma)
        else:
            nll = self.calculate_nll_(preds_test, y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(),
                                      sigma**2)
        print(mse)
        return nll

    def run_for_all_keys(self,pcl):
        all_keys = ['map_results', '1', '2', '5', '8', '14', '23', '37', '61', '100']
        nlls = []
        for key in all_keys:
            nlls.append(self.run_for_key(pcl, key))
        return nlls

    def run_for_multiple_files(self, paths):

        nlls = []
        pbar = tqdm(paths, desc='Running through files')
        for path in pbar:
            pcl = pickle.load(open(path, 'rb'))
            nlls.append(self.run_for_all_keys(pcl))
        return nlls

    def get_paths(self, path, criteria = None):
        if criteria is None:
            criteria = ""
        paths = [os.path.join(path, p) for p in os.listdir(path) if criteria in p]
        return paths

    def run_for_dataset(self, criteria= None):

        paths = self.get_paths(self.path_to_models, criteria)
        nlls = self.run_for_multiple_files(paths)
        return nlls