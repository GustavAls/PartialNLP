import pickle
import sys, os, time, requests
import argparse
import torch.nn as nn
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
import os
from partial_bnn_functional import *
from uci import UCIDataloader, UCIDataset, UCIBostonDataset, UCIEnergyDataset, UCIConcreteDataset, \
    UCIYachtDataset
from MAP_baseline.MapNN import MapNN


def calculate_nll_vi(model, dataloader, sigma, y_scale, y_loc, num_mc_runs=100, device='cpu'):
    for batch, label in dataloader:
        batch = batch.to(device)
        label = label.to(device)
        mc_matrix = torch.zeros((batch.shape[0], num_mc_runs))
        for mc_run in range(num_mc_runs):
            output = model(batch).detach().flatten()
            mc_matrix[:, mc_run] = output

        nll = calculate_nll_third(label, mc_matrix,sigma, y_scale, y_loc)
    return nll

def get_tau_by_conjugacy(x, alpha, beta):

    mean_x = torch.mean(x)
    n = len(x)
    dist = Gamma(alpha + n/2, beta + 1/2 * torch.sum((x - mean_x)**2))
    posterior_tau = torch.mean(dist.sample((1000,)))
    return posterior_tau

def calculate_nll_third(labels, mc_matrix, sigma, y_scale, y_loc):
    results = []
    for i in range(mc_matrix.shape[0]):
        res_temp = []
        for j in range(mc_matrix.shape[1]):
            dist = Normal(mc_matrix[i,j]*y_scale + y_loc, np.sqrt(sigma)*y_scale)
            res_temp.append(dist.log_prob(labels[i]*y_scale + y_loc).item())
        results.append(np.mean(res_temp))
    return np.mean(results)


def calculate_mse_vi(model, dataloader, num_mc_runs=600, device='cpu'):
    mc_overall = []
    for batch, label in dataloader:
        mc_batch = []
        batch = batch.to(device)
        label = label.to(device)
        for mc_run in range(num_mc_runs):
            output = model(batch)
            mc_batch.append((label - output) ** 2)

        mc_overall.append(torch.mean(torch.stack(mc_batch)).item())

    mse = np.mean(mc_overall)

    return mse


def calculate_sigma(predictions, labels, alpha = 3, beta = 1):
    residuals = np.array(predictions) - np.array(labels)
    residuals = torch.from_numpy(residuals)
    return get_tau_by_conjugacy(residuals,alpha,beta)


def make_multiple_runs_vi(dataset_class, data_path, num_runs, device='cpu', gap=True, train_args=None):
    for run in range(num_runs):
        dataset = dataset_class(data_dir=data_path,
                                test_split_type="gap" if gap else "random",
                                test_size=0.1,
                                gap_column='random',
                                val_fraction_of_train=0.1,
                                seed=np.random.randint(0, 1000))

        n_train, p = dataset.X_train.shape
        n_val = dataset.X_val.shape[0]
        out_dim = dataset.y_train.shape[1]
        n_test = dataset.X_test.shape[0]
        train_args['y_scale'] = dataset.scl_Y.scale_
        train_args['y_loc'] = dataset.scl_Y.mean_
        y_scale = dataset.scl_Y.scale_
        y_loc = dataset.scl_Y.mean_
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        percentages = [1, 61, 100]
        results = {
            'percentages': percentages,
            'x_train': dataset.X_train,
            'y_train': dataset.y_train,
            'y_val': dataset.y_val,
            'x_val': dataset.X_val,
            'x_test': dataset.X_test,
            'y_test': dataset.y_test,
            'val_nll': [],
            'test_nll': [],
            'val_mse': [],
            'test_mse': []

        }
        train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train)
        val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)
        test_dataloader = DataLoader(UCIDataloader(dataset.X_test, dataset.y_test), batch_size=n_test)

        results_dict = train_model_with_varying_stochasticity_scheme_two(MapNN(p, 50, out_dim, "leaky_relu"),
                                                                         train_dataloader,
                                                                         val_dataloader,
                                                                         percentages,
                                                                         train_args,
                                                                         run,
                                                                         dataloader_test=test_dataloader
                                                                         )

        results_dict_ordered = (
            results_dict['models'],
            results_dict['predictions_train'],
            results_dict['predictions_val'],
            results_dict['predictions_test']
        )

        for model, train_, val_, test_ in zip(*results_dict_ordered):
            sigma = calculate_sigma(*train_)
            sigma = 1/sigma
            results['val_nll'].append(calculate_nll_vi(
                model, val_dataloader, sigma, y_scale, y_loc,train_args['num_mc_samples'], device
            ))

            results['val_mse'].append(calculate_mse_vi(
                model, val_dataloader, train_args['num_mc_samples'], device
            ))

            results['test_nll'].append(calculate_nll_vi(
                model, val_dataloader, sigma, y_scale, y_loc, train_args['num_mc_samples'], device
            ))
            results['test_mse'].append(calculate_mse_vi(
                model, val_dataloader, train_args['num_mc_samples'], device
            ))

        save_name = os.path.join(
            train_args['save_path'], f'results_{run}_{dataset_class.dataset_name}.pkl'
        )
        with open(save_name, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(results['test_nll'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_epochs", type=int, default=20000)
    parser.add_argument("--dataset", type=str, default="boston")
    parser.add_argument('--data_path', type=str, default=os.getcwd())
    parser.add_argument("--gap", type=bool, default=False)
    parser.add_argument('--num_runs', type=int, default=15)
    parser.add_argument("--prior_variance", type=float,
                        default=0.1)  # 0.1 is good for yacht, but not for other datasets

    args = parser.parse_args()

    if args.dataset == "yacht":
        dataset_class = UCIYachtDataset
    elif args.dataset == "energy":
        dataset_class = UCIEnergyDataset
    elif args.dataset == "concrete":
        dataset_class = UCIConcreteDataset
    elif args.dataset == "boston":
        dataset_class = UCIBostonDataset
    else:
        dataset_class = UCIYachtDataset

    train_args = {
        'num_mc_samples': 200,
        'device': args.device,
        'epochs': args.num_epochs,
        'save_path': args.output_path

    }
    make_multiple_runs_vi(dataset_class, args.data_path, args.num_runs, args.device, args.gap, train_args)
