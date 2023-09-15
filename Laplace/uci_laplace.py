import sys, os, time, requests
import argparse
import numpy as np
import pandas as pd
import torch.distributions
from MAP_baseline import trainer
from MAP_baseline.MapNN import MapNN
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.distributions import Normal
from laplace import Laplace
from laplace.utils import LargestMagnitudeSubnetMask
from uci import UCIDataloader, UCIDataset, UCIBostonDataset, UCIEnergyDataset, UCIYachtDataset
from torch.nn import MSELoss


def count_parameters(model):
    """Count the number of parameters in a model.
        Args:
            model: (nn.Module) model
        Returns:
            num_params: (int) number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def calculate_mse(preds, labels, var, num_mc_runs=600):
    """Calculate the mean squared error of the predictions.
        Args:
            preds: (np.array) predictions of the model
            label: (np.array) true labels
        Returns:
            mse: (float) mean squared error
    """
    mse_loss = MSELoss()
    results = []
    for (pred, label) in zip(preds, labels):
        mse_batch = []
        dist = Normal(pred, np.sqrt(var))
        samples = dist.sample((num_mc_runs,))
        for sample in samples:
            mse_batch.append(mse_loss(sample, label))
        results.append(np.mean(mse_batch))
    mse = np.mean(results)
    return mse


def calculate_nll(preds, labels, var, y_scale, y_loc, num_mc_runs=600):
    """Calculate the negative log likelihood of the predictions.
        Args:
            preds: (np.array) predictions of the model
            label: (np.array) true labels
            sigma: (float) variance of the predictions
        Returns:
            nll: (float) negative log likelihood
    """
    results = []
    for (pred, label) in zip(preds, labels):
        log_likelihood = []
        for mc_run in range(num_mc_runs):
            dist = Normal((y_scale * pred) + y_loc, torch.sqrt(var) * y_scale)
            log_prob = dist.log_prob(y_loc + (label * y_scale))
            log_likelihood.append(log_prob)
        results.append(torch.mean(torch.cat(log_likelihood, dim=0)).item())

    nll = -np.mean(results)
    return nll


def run_percentiles(mle_model, train_dataloader, dataset, percentages):
    """Run the Laplace approximation for different subnetworks.
        Args:
            mle_model: (nn.Module) trained model
            train_dataloader: (torch.utils.data.DataLoader) dataloader for the training data
            dataset: (UCIDataset) dataset
            percentages: (list) list of percentages of the parameters to use
        Returns:
            val_nll: (list) list of validation negative log likelihoods
            test_nll: (list) list of test negative log likelihoods
            val_mse: (list) list of validation mean squared errors
            test_mse: (list) list of test mean squared errors
    """
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

        y_scale = torch.from_numpy(dataset.scl_Y.scale_)
        y_loc = torch.from_numpy(dataset.scl_Y.mean_)

        val_targets = torch.from_numpy(dataset.y_val).to(torch.float32)
        val_pred_mu, val_pred_var = la(torch.from_numpy(dataset.X_val).to(torch.float32))

        test_targets = torch.from_numpy(dataset.y_test).to(torch.float32)
        test_pred_mu, test_pred_var = la(torch.from_numpy(dataset.X_test).to(torch.float32))

        val_mse.append(calculate_mse(val_pred_mu, val_targets, torch.mean(val_pred_var), 600))
        val_nll.append(calculate_nll(val_pred_mu, val_targets, torch.mean(val_pred_var), y_scale, y_loc))
        test_mse.append(calculate_mse(test_pred_mu, test_targets, torch.mean(test_pred_var), 600))
        test_nll.append(calculate_nll(test_pred_mu, test_targets, torch.mean(test_pred_var), y_scale, y_loc))
        print("Percentile: ", p, "Val MSE: ", val_mse[-1], "Test MSE: ", test_mse[-1])
        print("Percentile: ", p, "Val NLL: ", val_nll[-1], "Test NLL: ", test_nll[-1])
    return val_nll, test_nll, val_mse, test_mse


def multiple_runs(data_path, dataset_class, num_runs, device, num_epochs, output_path):
    """Run the Laplace approximation for different subnetworks.
        Args:
            data_path: (str) path to the data
            dataset_class: (UCIDataset) dataset class
            num_runs: (int) number of runs
            device: (str) device to use
            num_epochs: (int) number of epochs to train
            output_path: (str) path to save the results
    """
    for run in range(num_runs):
        # percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
        percentages = [23, 37, 61, 100]
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

    print("Laplace experiments finished!")


