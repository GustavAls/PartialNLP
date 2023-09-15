import copy

import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from torch.utils.data import DataLoader
from swag_temp import run_swag_partial
from PartialNLP.VI.partial_bnn_functional import create_mask, train, create_non_parameter_mask
from PartialNLP.MAP_baseline.MapNN import MapNN
from PartialNLP.uci import UCIDataloader, UCIDataset, UCIBostonDataset, UCIEnergyDataset, UCIConcreteDataset, \
    UCIYachtDataset
from torch.distributions.multivariate_normal import MultivariateNormal
import argparse


def parameters_to_vector(parameters) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        vec.append(param.view(-1))
    return torch.cat(vec)


def calculate_nll(preds, labels, sigma):
    results = []
    for pred, label in zip(preds, labels):
        dist = MultivariateNormal(pred.ravel(), torch.eye(pred.shape[0]) * sigma)
        results.append(dist.log_prob(label.ravel()).item())

    nll = -np.mean(results)
    return nll


def calculate_empirical_var(preds, labels):
    results = []
    for pred, label in zip(preds, labels):
        results.append(pred - label)
    empirical_variance = torch.var(torch.cat(results, dim = 0)).item()
    return empirical_variance


def calculate_mse(preds, labels):

    results = []
    for pred, label in zip(preds, labels):
        results.append((pred-label)**2)
    mse = torch.mean(torch.cat(results, dim = 0)).item()
    return mse


def evaluate_swag(model,dataloader, mask, swag_results, train_args, sigma = 1):

    theta_swa = swag_results["theta_swa"]
    sigma_diag = swag_results["sigma_diag"]
    D = swag_results["D"]
    K = swag_results["K"]

    model_ = copy.deepcopy(model)
    true_model_params = nn.utils.parameters_to_vector(model.parameters())
    num_mc_samples = train_args['num_mc_samples']
    device = train_args['device']

    mc_overall_preds = []
    mc_overall_labels = []
    for batch, labels in dataloader:
        batch, labels = batch.to(device), labels.to(device)
        mc_batch = []
        mc_labels = []
        for mc_run in range(num_mc_samples):
            z1 = torch.normal(mean=torch.zeros((theta_swa.numel())), std=1.0).to(device)
            z2 = torch.normal(mean=torch.zeros(K), std=1.0).to(device)

            theta = theta_swa + 2 ** -0.5 * (sigma_diag ** 0.5 * z1) + (2 * (K - 1)) ** -0.5 * (
                            D @ z2[:, None]).flatten()

            true_model_params[mask] = theta
            nn.utils.vector_to_parameters(true_model_params, model_.parameters())
            mc_batch.append(model_(batch))
            mc_labels.append(labels)

        mc_overall_preds.append(mc_batch)
        mc_overall_labels.append(mc_labels)

    mc_overall_preds = [x for y in mc_overall_preds for x in y]
    mc_overall_labels = [x for y in mc_overall_labels for x in y]

    nll = calculate_nll(mc_overall_preds, mc_overall_labels, sigma)
    mse = calculate_mse(mc_overall_preds, mc_overall_labels)
    empirical_var = calculate_empirical_var(mc_overall_preds, mc_overall_labels)
    return nll, mse, empirical_var



def train_swag(untrained_model, dataloader, dataloader_val, dataloader_test, percentages, train_args):

    model_ = train(
        copy.deepcopy(untrained_model),
        dataloader,
        dataloader_val,
        model_old=None,
        vi=False,
        device=train_args['device'],
        epochs=train_args['epochs'],
        save_path=None
    )

    learning_rate_sweep = train_args['learning_rate_sweep']
    results_dict = {
        'nll_test': [],
        'mse_test': [],
        'best_nll_val': [],
        'according_mse_val': []
    }

    for percentage in percentages:
        mask = create_non_parameter_mask(model_, percentage)
        mask = mask.bool()
        model = copy.deepcopy(model_)
        nlls = []
        mses = []
        for lr in learning_rate_sweep:
            swag_results = run_swag_partial(model, dataloader, lr, n_epochs=50, criterion=nn.MSELoss, mask=mask)
            nll, mse, sigma = evaluate_swag(model,dataloader,mask, swag_results, train_args)
            nll, mse, sigma = evaluate_swag(model, dataloader_val, mask, swag_results, train_args, sigma = sigma)
            nlls.append(nll)
            mses.append(mse)

        print("Best Validation MSE for percentage", percentage, 'was', np.min(nll))
        lr = learning_rate_sweep[np.argmin(nlls)]
        swag_results = run_swag_partial(model, dataloader, lr, n_epochs=30, criterion=nn.MSELoss, mask=mask)
        nll, mse, sigma = evaluate_swag(model, dataloader, mask, swag_results, train_args)
        nll, mse, sigma = evaluate_swag(model, dataloader_test, mask, swag_results, train_args, sigma=sigma)
        results_dict['nll_test'].append(nll)
        results_dict['mse_test'].append(mse)
        results_dict['best_nll_val'].append(np.min(nlls))
        results_dict['according_mse_val'].append(mses[np.argmin(nlls)])

    return results_dict


def make_multiple_runs_swag(dataset_class, data_path, num_runs, device='cpu', gap=True, train_args=None):

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
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]

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

        train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train//8)
        val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)
        test_dataloader = DataLoader(UCIDataloader(dataset.X_test, dataset.y_test), batch_size=n_test)
        untrained_model = MapNN(p, 35, 2, out_dim, "leaky_relu")

        res = train_swag(
            untrained_model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            percentages,
            train_args
        )
        results['val_nll'] = res['best_nll_val']
        results['val_mse'] = res['according_mse_val']
        results['test_nll'] = res['nll_test']
        results['test_mse'] = res['mse_test']

        save_name = os.path.join(train_args['save_path'],
                                 f'results_swag_run_{run}.pkl')
        with open(save_name, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        'num_mc_samples': 600,
        'device': args.device,
        'epochs': args.num_epochs,
        'save_path': args.output_path,
        'learning_rate_sweep': np.logspace(-5, -2, 10, endpoint=True)[::-1]
    }

    make_multiple_runs_swag(dataset_class, args.data_path, args.num_runs, args.device, args.gap, train_args)
