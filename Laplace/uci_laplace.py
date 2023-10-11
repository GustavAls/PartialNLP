import ast
import copy
import sys, os, time, requests
import argparse
import numpy as np
import pandas as pd
import torch.distributions
import tqdm
from SWAG.swag_uci import train_swag
from SWAG.swag_temp import *
from MAP_baseline.MapNN import MapNN, MapNNRamping
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.distributions import Normal
from laplace import Laplace
from laplace.utils import LargestMagnitudeSubnetMask
from uci import UCIDataloader, UCIDataset, UCIBostonDataset, UCIEnergyDataset, UCIYachtDataset, UCIWineDataset
from torch.nn import MSELoss
from torch.distributions import Gamma
import misc.likelihood_losses as ll_losses
from VI.partial_bnn_functional import train

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
    scales = var
    for pred, scale, label in zip(preds, scales, labels):
        dist = Normal(pred * y_scale + y_loc, scale * y_scale)
        results.append(dist.log_prob(label * y_scale + y_loc))
    nll = -1 * sum(results) / len(results)
    return nll


def calculate_precision_from_prior(residuals, alpha=3, beta=5):
    """

    :param residuals: torch.tensor of form (predictions - labels)
    :param alpha: (float) Parameter in gamma distribution
    :param beta: (float) Parameter in gamma distribution
    :return: mean of the conjugate distribution p(tau | x) ~ Gamma()
    """

    residuals = residuals.flatten()
    mean_r = 0
    n = len(residuals)
    dist = Gamma(alpha + n / 2, beta + 1 / 2 * torch.sum((residuals - mean_r) ** 2))
    posterior_tau = torch.mean(dist.sample((1000,)))
    return posterior_tau


def calculate_variance(model, dataloader, alpha=3, beta=5, beta_prior = True):
    residuals = []
    for batch, label in dataloader:
        try:
            output, _ = model(batch)
        except:
            output = model(batch)

        residuals.append(output - label)
    residuals = torch.cat(residuals, dim = 0)

    if beta_prior:
        precision = calculate_precision_from_prior(residuals, alpha, beta)
        sigma = 1 / torch.sqrt(precision)
    else:
        sigma = residuals.detach().std()

    return sigma

def predict(val_data, model):
    return model(val_data)


def calculate_nll_potential_final(f_mu, f_std, labels, sigma, num_samples=100, y_scale=1, y_loc=1):
    results = []
    for pred, lab in zip(f_mu, labels):
        dist_sampler = Normal(f_mu, f_std)
        results_tmp = []
        for sample in dist_sampler.sample((num_samples,)):
            d = Normal(sample, sigma)
            results_tmp.append(d.log_prob(lab))

        results.append(np.mean(results_tmp))
    return -1 * np.mean(results)


def find_best_prior(ml_model, subnetwork_indices, train_dataloader, val_loader, y_scale, y_loc, sigma_noise = 0):
    prior_precision_sweep = np.logspace(-3, 5, num=20, endpoint=True)
    batch, label = next(iter(val_loader))
    results = []
    for prior in tqdm(prior_precision_sweep):
        la = Laplace(copy.deepcopy(ml_model), 'regression',
                     subset_of_weights='subnetwork',
                     hessian_structure='full',
                     subnetwork_indices=subnetwork_indices,
                     prior_precision=torch.tensor([float(prior)]))

        la.fit(train_dataloader)
        f_mu, f_var = la(batch)
        f_var = f_var.detach().squeeze().sqrt()
        results.append(calculate_nll(f_mu.flatten(),
                                     label.flatten(), torch.sqrt(f_var**2 + sigma_noise**2), y_scale, y_loc))

    return prior_precision_sweep[np.argmin(results)]


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
    y_scale = torch.from_numpy(dataset.scl_Y.scale_)
    y_loc = torch.from_numpy(dataset.scl_Y.mean_)
    # Base case needed, for when find_best_prior does not converge
    best_prior = 1.0
    for p in percentages:
        ml_model = copy.deepcopy(mle_model)
        subnetwork_mask = LargestMagnitudeSubnetMask(ml_model, n_params_subnet=int((p / 100) * num_params))
        subnetwork_indices = subnetwork_mask.select()
        batch, labels = next(iter(train_dataloader))

        sigma_noise = calculate_variance(ml_model, train_dataloader, alpha=3, beta=1, beta_prior = False)

        best_prior = find_best_prior(mle_model, subnetwork_indices, train_dataloader,
                                     DataLoader(UCIDataloader(dataset.X_val, dataset.y_val, ),
                                                batch_size=dataset.X_val.shape[0]),
                                     y_scale=y_scale, y_loc=y_loc, sigma_noise = sigma_noise)


        # Define and fit subnetwork LA using the specified subnetwork indices
        la = Laplace(ml_model, 'regression',
                     subset_of_weights='subnetwork',
                     hessian_structure='full',
                     subnetwork_indices=subnetwork_indices,
                     prior_precision=torch.tensor([float(best_prior)]))

        print('Best prior was', best_prior)
        la.fit(train_dataloader)

        val_targets = torch.from_numpy(dataset.y_val).to(torch.float32)
        val_preds_mu, val_preds_var = la(torch.from_numpy(dataset.X_val).to(torch.float32))
        val_preds_sigma = val_preds_var.squeeze().sqrt()
        predictive_std_val = val_preds_sigma

        test_targets = torch.from_numpy(dataset.y_test).to(torch.float32)
        test_preds_mu, test_preds_var = la(torch.from_numpy(dataset.X_test).to(torch.float32))
        test_preds_sigma = test_preds_var.squeeze().sqrt()
        predictive_std_test = test_preds_sigma
        val_targets = val_targets.flatten()
        test_targets = test_targets.flatten()
        if p == percentages[0]:
            val_mse.append(calculate_mse(val_preds_mu.flatten(), val_targets))
            val_nll.append(calculate_nll(
                val_preds_mu.flatten(), val_targets, torch.tile(torch.sqrt(sigma_noise), (len(val_targets),)), y_scale, y_loc))
            test_mse.append(calculate_mse(val_preds_mu.flatten(), val_targets))
            test_nll.append(calculate_nll(
                test_preds_mu.flatten(), test_targets, torch.tile(torch.sqrt(sigma_noise), (len(test_targets),)), y_scale, y_loc))

        pred_std_val = torch.sqrt(predictive_std_val**2 + sigma_noise**2)
        pred_std_test = torch.sqrt(predictive_std_test**2 + sigma_noise**2)
        val_mse.append(calculate_mse(val_preds_mu.flatten(), val_targets))
        val_nll.append(calculate_nll(val_preds_mu.flatten(), val_targets, pred_std_val, y_scale, y_loc))
        test_mse.append(calculate_mse(test_preds_mu.flatten(), test_targets))
        test_nll.append(calculate_nll(test_preds_mu.flatten(), test_targets, pred_std_test, y_scale, y_loc))
        print("Percentile: ", p, "Val MSE: ", val_mse[-1], "Test MSE: ", test_mse[-1])
        print("Percentile: ", p, "Val NLL: ", val_nll[-1], "Test NLL: ", test_nll[-1])
    return val_nll, test_nll, val_mse, test_mse


def multiple_runs(data_path, dataset_class, num_runs, device, num_epochs, output_path, fit_swag, **kwargs):
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
    tmp_res = []
    for run in range(num_runs):
        percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
        dataset = dataset_class(data_dir=data_path,
                                test_split_type='random',
                                test_size=0.1,
                                seed=np.random.randint(0, 1000),
                                val_fraction_of_train=0.1)

        n_train, p = dataset.X_train.shape
        n_val = dataset.X_val.shape[0]
        n_test = dataset.X_test.shape[0]
        out_dim = dataset.y_train.shape[1]

        mle_model = MapNN(input_size=p, width=50, output_size=out_dim, non_linearity="leaky_relu")

        if issubclass(loss := kwargs.get('loss', MSELoss), ll_losses.BaseMAPLossSwag):
            kwargs['model'] = mle_model
            loss_fn = loss(**kwargs)
        else:
            loss_fn = loss()

        train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train//8)
        val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)
        test_dataloader = DataLoader(UCIDataloader(dataset.X_test, dataset.y_test), batch_size=n_test)

        mle_model = train(network=mle_model, dataloader_train=train_dataloader, dataloader_val=val_dataloader,
                             model_old = None, vi = False, device='cpu', epochs = num_epochs,
                             save_path = output_path, return_best_model=True, criterion=loss_fn)
        val_nll, test_nll, val_mse, test_mse = run_percentiles(mle_model, train_dataloader, dataset, percentages)

        save_name = os.path.join(output_path, f'results_laplace_run_{run}.pkl')
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

        if fit_swag:
            results_swag = {
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

            loss_arguments = {'loss': ll.GLLGP_loss_swag, 'prior_sigma': 1 / args.prior_precision}
            loss = loss_arguments.get('loss', MSELoss)
            loss_fn = loss(**loss_arguments)

            train_args = { 'bayes_var': False,
                           'y_scale': dataset.scl_Y.scale_,
                           'y_loc': dataset.scl_Y.mean_ ,
                           'loss': loss_fn,
                           'num_mc_samples': 100,
                           'device': args.device,
                           'epochs': args.num_epochs,
                           'save_path': args.output_path,
                           'learning_rate_sweep': np.logspace(-5, -1, 10, endpoint=True),
                           'swag_epochs': 100,
                           }

            res = train_swag(mle_model, train_dataloader, val_dataloader, test_dataloader,
                             percentages, trained_model=mle_model, train_args=train_args)
            results_swag['val_nll'] += res['best_nll_val']
            results_swag['val_mse'] += res['according_mse_val']
            results_swag['test_nll'] += res['nll_test']
            results_swag['test_mse'] += res['mse_test']

            save_name = os.path.join(output_path, f'results_swag_run_{run}.pkl')
            with open(save_name, 'wb') as handle:
                pickle.dump(results_swag, handle, protocol=pickle.HIGHEST_PROTOCOL)

        all_res.append(results)

    return all_res


def make_size_ramping(data_path, dataset_class, num_runs, device, num_epochs, output_path, **kwargs):
    all_res = []
    widths = [i for i in range(30, 110, 10)]
    depths = [1, 2, 3]
    percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
    for run in range(num_runs):
        dataset = dataset_class(data_dir=data_path,
                                test_split_type='random',
                                test_size=0.1,
                                seed=np.random.randint(0, 1000),
                                val_fraction_of_train=0.1)
        n_train, p = dataset.X_train.shape
        n_val = dataset.X_val.shape[0]
        out_dim = dataset.y_train.shape[1]
        train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train)
        val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)
        for depth in depths:
            for width in widths:
                mle_model = MapNNRamping(
                    input_size=p,
                    width=width,
                    output_size=out_dim,
                    num_hidden_layers=depth,
                    non_linearity="leaky_relu",
                )

                if issubclass(loss := kwargs.get('loss_fn', MSELoss), ll_losses.BaseMAPLoss):
                    kwargs['model'] = mle_model
                    loss_fn = loss(**kwargs)
                else:
                    loss_fn = loss()

                train_args = {
                    'device': device,
                    'epochs': num_epochs,
                    'save_path': output_path,
                    'early_stopping_patience': 1000,
                    'loss_fn': loss_fn
                }

                mle_model = train(network=mle_model, dataloader_train=train_dataloader, dataloader_val=val_dataloader,
                                  **train_args)

                val_nll, test_nll, val_mse, test_mse = run_percentiles(mle_model, train_dataloader, dataset,
                                                                       percentages)

                save_name = os.path.join(output_path,
                                         f'results_{run}_depth_{depth}_width_{width}_{dataset_class.dataset_name}.pkl')

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
                    'test_mse': test_mse,
                    'run': run,
                    'width': width,
                    'depth': depth
                }
                # with open(save_name, 'wb') as handle:
                #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("nll for percentages", percentages, 'for width', width, 'and depth', depth, 'was ', test_nll)
                all_res.append(results)
    return all_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_epochs", type=int, default=20000)
    parser.add_argument("--dataset", type=str, default="boston")
    parser.add_argument('--data_path', type=str, default=os.getcwd())
    parser.add_argument('--num_runs', type=int, default=15)
    parser.add_argument('--size_ramping', type=ast.literal_eval, default=False)
    parser.add_argument('--get_map', type=ast.literal_eval, default=True)
    parser.add_argument('--fit_swag', type=ast.literal_eval, default=True)
    parser.add_argument('--prior_precision', type=float, default=0.5)

    # TODO implement initialisation corresponding to prior precision

    args = parser.parse_args()

    if args.dataset == "yacht":
        dataset_class = UCIYachtDataset
    elif args.dataset == "energy":
        dataset_class = UCIEnergyDataset
    elif args.dataset == "boston":
        dataset_class = UCIBostonDataset
    elif args.dataset == 'wine':
        dataset_class = UCIWineDataset
    else:
        dataset_class = UCIYachtDataset

    loss_arguments = {'loss': ll_losses.GLLGP_loss_swag if args.get_map else MSELoss,
                      'prior_sigma': 1/args.prior_precision}

    if args.size_ramping:
        results = make_size_ramping(args.data_path, dataset_class, args.num_runs, args.device, args.num_epochs,
                                    args.output_path, **loss_arguments)
        with open(os.path.join(args.output_path, f'results_ramping_{args.dataset}.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        results = multiple_runs(args.data_path, dataset_class, args.num_runs, args.device, args.num_epochs,
                                args.output_path, args.fit_swag, **loss_arguments)

    print("Laplace experiments finished!")
