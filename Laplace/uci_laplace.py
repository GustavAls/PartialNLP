import sys, os, time, requests
import argparse
import numpy as np
import pandas as pd
import torch.distributions
from MAP_baseline import trainer
from MAP_baseline.MapNN import MapNN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, Trace_ELBO, autoguide, SVI
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from laplace import Laplace
from laplace.utils import LargestMagnitudeSubnetMask
from uci import UCIDataloader, UCIDataset, UCIBostonDataset, UCIEnergyDataset, UCIYachtDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_nll(preds, labels, sigma):
    results = []
    for pred, label in zip(preds, labels):
        dist = MultivariateNormal(pred.ravel(), torch.eye(pred.shape[0]) * sigma)
        results.append(dist.log_prob(label.ravel()).item())

    nll = -np.mean(results)
    return nll

def calculate_mse(preds, labels):

    results = []
    for pred, label in zip(preds, labels):
        results.append((pred-label)**2)
    mse = torch.mean(torch.cat(results, dim = 0)).item()
    return mse

def run_percentiles(mle_model, train_dataloader, dataset, percentages):
    num_params = count_parameters(mle_model)
    val_nll = []
    test_nll = []
    val_mse = []
    test_mse = []
    for p in percentages:
        subnetwork_mask = LargestMagnitudeSubnetMask(mle_model, n_params_subnet=int((p / 100) * num_params))
        subnetwork_indices = subnetwork_mask.select()

        # Define and fit subnetwork LA using the specified subnetwork indices
        la = Laplace(mle_model, 'regression',
                     subset_of_weights='subnetwork',
                     hessian_structure='full',
                     subnetwork_indices=subnetwork_indices)
        la.fit(train_dataloader)
        targets = torch.from_numpy(dataset.y_val).to(torch.float32)
        f_mu, f_var = la(torch.from_numpy(dataset.X_test).to(torch.float32))
        print("f_mu")

    return val_nll, test_nll, val_mse, test_mse


def multiple_runs(data_path, dataset_class, num_runs, device, num_epochs, output_path):
    for run in range(num_runs):
        percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
        dataset = dataset_class(data_dir=data_path,
                                test_split_type='random',
                                test_size=0.1,
                                seed=np.random.randint(0, 1000),
                                val_fraction_of_train=0.1)

        n_train, p = dataset.X_train.shape
        n_val = dataset.X_val.shape[0]
        out_dim = dataset.y_train.shape[1]

        mle_model = MapNN(input_size=p, width=50, output_size=out_dim, non_linearity="leaky_relu")
        train_args = {
            'device': device,
            'epochs': num_epochs,
            'save_path': output_path,
            'early_stopping_patience': 1000
        }
        train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train)
        val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)

        trainer.train(network=mle_model, dataloader_train=train_dataloader, dataloader_val=val_dataloader, **train_args)
        val_nll, test_nll, val_mse, test_mse = run_percentiles(mle_model, train_dataloader, dataset, percentages)

        save_name = os.path.join(output_path, f'results_{run}_{dataset_class.dataset_name}.pkl')
        results = {
            'percentages': percentages,
            'x_train': dataset.X_train,
            'y_train': dataset.y_train,
            'y_val': dataset.y_val,
            'x_val': dataset.X_val,
            'x_test': dataset.X_test,
            'y_test': dataset.y_test,
            'val_nll': val_nll,
            'test_nll': test_nll,
            'val_mse': val_mse,
            'test_mse': test_mse
        }
        with open(save_name, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_epochs", type=int, default=20000)
    parser.add_argument("--dataset", type=str, default="boston")
    parser.add_argument('--data_path', type=str, default=os.getcwd())
    parser.add_argument('--num_runs', type=int, default=15)

    args = parser.parse_args()

    if args.dataset == "yacht":
        dataset_class = UCIYachtDataset
    elif args.dataset == "energy":
        dataset_class = UCIEnergyDataset
    elif args.dataset == "boston":
        dataset_class = UCIBostonDataset
    else:
        dataset_class = UCIYachtDataset

    multiple_runs(args.data_path, dataset_class, args.num_runs, args.device, args.num_epochs, args.output_path)

    # Examples of different ways to specify the subnetwork
    # via indices of the vectorized model parameters
    # Example 1: select the 128 parameters with the largest magnitude
    # Hardcoded

    print("Computed laplace preds")


