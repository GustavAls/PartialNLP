import copy
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
from torch.distributions import Gamma
from laplace.curvature import AsdlGGN
def count_parameters(model):
    """Count the number of parameters in a model.
        Args:
            model: (nn.Module) model
        Returns:
            num_params: (int) number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def calculate_mse(preds, labels):
    """Calculate the mean squared error of the predictions.
        Args:
            preds: (np.array) predictions of the model
            label: (np.array) true labels
        Returns:
            mse: (float) mean squared error
    """
    mse_loss = MSELoss()
    base_pred = mse_loss(preds, labels)
    print("Loss using means is equal to", base_pred)
    return base_pred

def calculate_nll(preds, labels, var, y_scale, y_loc):
    """Calculate the negative log likelihood of the predictions.
        Args:
            preds: (np.array) predictions of the model
            label: (np.array) true labels
            sigma: (float) variance of the predictions
        Returns:
            nll: (float) negative log likelihood
    """
    results = []
    scales = torch.sqrt(var)
    for pred, scale, label in zip(preds, scales, labels):
        dist = Normal(pred*y_scale + y_loc, scale*y_scale)
        results.append(dist.log_prob(label))
    nll = -1 * np.mean(results)
    return nll

def calculate_precision_from_prior(residuals, alpha = 3, beta = 5):
    """

    :param residuals: torch.tensor of form (predictions - labels)
    :param alpha: (float) Parameter in gamma distribution
    :param beta: (float) Parameter in gamma distribution
    :return: mean of the conjugate distribution p(tau | x) ~ Gamma()
    """

    residuals = residuals.flatten()
    mean_r = torch.mean(residuals)
    n = len(residuals)
    dist = Gamma(alpha + n/2, beta + 1/2 * torch.sum((residuals - mean_r)))
    posterior_tau = torch.mean(dist.sample((1000,)))
    return posterior_tau

def calculate_variance(model, batch, labels, alpha = 3, beta = 5):
    preds, _ = model(batch)

    residuals = preds.flatten() - labels.flatten()

    precision = calculate_precision_from_prior(residuals, alpha, beta)
    variance = 1/precision
    return variance

def predict(val_data, model):
    return model(val_data)



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
        ml_model = copy.deepcopy(mle_model)
        subnetwork_mask = LargestMagnitudeSubnetMask(ml_model, n_params_subnet=int((p / 100) * num_params))
        subnetwork_indices = subnetwork_mask.select()

        # Define and fit subnetwork LA using the specified subnetwork indices
        la = Laplace(ml_model, 'regression',
                     subset_of_weights='subnetwork',
                     hessian_structure='full',
                     subnetwork_indices=subnetwork_indices)
        la.fit(train_dataloader)

        y_scale = torch.from_numpy(dataset.scl_Y.scale_)
        y_loc = torch.from_numpy(dataset.scl_Y.mean_)
        batch, labels = next(iter(train_dataloader))

        sigma_noise = calculate_variance(la, batch, labels, alpha = 3, beta = 1)
        val_targets = torch.from_numpy(dataset.y_val).to(torch.float32)
        val_preds_mu, val_preds_var = la(torch.from_numpy(dataset.X_val).to(torch.float32))
        val_preds_sigma = val_preds_var.squeeze().sqrt()
        predictive_std_val = torch.sqrt(val_preds_sigma**2 + sigma_noise)

        test_targets = torch.from_numpy(dataset.y_test).to(torch.float32)
        test_preds_mu, test_preds_var = la(torch.from_numpy(dataset.X_test).to(torch.float32))
        test_preds_sigma = test_preds_var.squeeze().sqrt()
        predictive_std_test = torch.sqrt(test_preds_sigma ** 2 + sigma_noise)
        val_targets = val_targets.flatten()
        test_targets = test_targets.flatten()

        val_mse.append(calculate_mse(val_preds_mu.flatten(), val_targets))
        val_nll.append(calculate_nll(val_preds_mu.flatten(), val_targets, predictive_std_val, y_scale, y_loc))
        test_mse.append(calculate_mse(test_preds_mu.flatten(), test_targets))
        test_nll.append(calculate_nll(test_preds_mu.flatten(), test_targets, predictive_std_test, y_scale, y_loc))
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
    all_res = []
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

        mle_model = trainer.train(network=mle_model, dataloader_train=train_dataloader, dataloader_val=val_dataloader, **train_args)
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

        all_res.append(results)
    return all_res

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

    results = multiple_runs(args.data_path, dataset_class, args.num_runs, args.device, args.num_epochs, args.output_path)
    breakpoint()
    print("Laplace experiments finished!")


