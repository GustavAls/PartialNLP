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
from MAP_baseline.MapNN import MapNN, MapNNRamping, MAPNNLike
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.distributions import Normal
from Laplace.laplace import Laplace
from Laplace.laplace.utils import LargestMagnitudeSubnetMask, RandomSubnetMask
from uci import UCIDataloader, UCIDataset, UCIBostonDataset, UCIEnergyDataset, UCIYachtDataset, UCIWineDataset
from torch.nn import MSELoss
from torch.distributions import Gamma
import misc.likelihood_losses as ll_losses
from VI.partial_bnn_functional import train
from HMC.uci_hmc import PredictiveHelper

### NOT CURRENTLY USED
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


def calculate_nll(preds, labels, sigma, y_scale, y_loc):
    """Calculate the negative log likelihood of the predictions.
        Args:
            preds: (np.array) predictions of the model
            label: (np.array) true labels
            sigma: (float) variance of the predictions
        Returns:
            nll: (float) negative log likelihood
    """
    results = []
    scales = sigma
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


def calculate_std(model, dataloader, alpha=3, beta=5, beta_prior = True):
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

def find_best_prior_kron(ml_model, train_dataloader, val_loader, y_scale, y_loc, sigma_noise = 0):
    prior_precision_sweep = np.logspace(-3, 5, num=20, endpoint=True)
    batch, label = next(iter(val_loader))
    results = []
    for prior in tqdm(prior_precision_sweep):
        la = Laplace(copy.deepcopy(ml_model), 'regression',
                     subset_of_weights='all',
                     hessian_structure='kron',
                     prior_precision=torch.tensor([float(prior)]))

        la.fit(train_dataloader)
        f_mu, f_var = la(batch)
        f_var = f_var.detach().squeeze().sqrt()
        results.append(calculate_nll(f_mu.flatten(),
                                     label.flatten(), torch.sqrt(f_var**2 + sigma_noise**2), y_scale, y_loc))

    return prior_precision_sweep[np.argmin(results)]

def compute_metrics(train, val, test, dataset, test_mu=None):
    tp = PredictiveHelper("")
    test += (test_mu - test.mean(1).reshape(-1, 1))
    fmu, fvar = tp.glm_predictive(test, std=True)
    nll_glm = tp.glm_nll(fmu.reshape((-1,)), fvar.reshape((-1,)), dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item())
    residuals = tp.get_residuals(train, dataset.y_train, full=True)
    res_test = tp.get_residuals(test, dataset.y_test, full=True)
    sigma = tp.get_sigma(residuals.mean(1))
    elpd = tp.calculate_nll_(test, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(), sigma ** 2)
    elpd_sqrt = tp.calculate_nll_(test, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(), sigma)
    return nll_glm, elpd, elpd_sqrt


def run_percentiles(mle_model, train_dataloader, dataset, percentages, save_name, random_mask = False,
                    largest_variance_mask = False):
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

    y_scale = torch.from_numpy(dataset.scl_Y.scale_)
    y_loc = torch.from_numpy(dataset.scl_Y.mean_)
    # Base case needed, for when find_best_prior does not converge
    best_prior = 1.0
    ph = PredictiveHelper("")

    laplace_result_dict =  pickle.load(open(save_name, "rb")) if os.path.exists(save_name) else None

    for p in percentages:
        if laplace_result_dict is None or str(p) not in laplace_result_dict.keys():
            ml_model = copy.deepcopy(mle_model)
            if random_mask:
                subnetwork_mask = RandomSubnetMask(ml_model, n_params_subnet=int((p / 100) * num_params))
            else:
                subnetwork_mask = LargestMagnitudeSubnetMask(ml_model, n_params_subnet=int((p / 100) * num_params))

            if largest_variance_mask:
                la = Laplace(ml_model, 'regression',
                             subset_of_weights='all',
                             hessian_structure='full')

                la.fit(train_dataloader)
                posterior_scale = la.posterior_precision.diagonal()**(-0.5)
                proxy_model = copy.deepcopy(ml_model)
                nn.utils.vector_to_parameters(posterior_scale, proxy_model.parameters())
                subnetwork_mask = LargestMagnitudeSubnetMask(proxy_model, n_params_subnet=int((p / 100) * num_params))

            subnetwork_indices = subnetwork_mask.select()
            batch, labels = next(iter(train_dataloader))

            sigma_noise = calculate_std(ml_model, train_dataloader, alpha=3, beta=1, beta_prior = False)

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

            test_targets = torch.from_numpy(dataset.y_test).to(torch.float32)

            train_preds_mu, train_preds_var = la(torch.from_numpy(dataset.X_train).to(torch.float32))
            val_preds_mu, val_preds_var = la(torch.from_numpy(dataset.X_val).to(torch.float32))
            test_preds_mu, test_preds_var = la(torch.from_numpy(dataset.X_test).to(torch.float32))

            if p == percentages[0]:
                nll_map_sqrt = ph.calculate_nll_(test_preds_mu.numpy(), dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(), torch.sqrt(sigma_noise))
                nll_map = ph.calculate_nll_(test_preds_mu.numpy(), dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(), sigma_noise)
                laplace_result_dict = {
                    'dataset': dataset,
                    'map_results': {'map_params': mle_model.state_dict(),
                                    'predictive_train': train_preds_mu,
                                    'predictive_val': val_preds_mu,
                                    'predictive_test': test_preds_mu,
                                    'glm_nll': nll_map,
                                    'elpd': nll_map,
                                    'elpd_spurious_sqrt': nll_map_sqrt,
                                    'elpd_gamma_prior': nll_map,
                                    'prior_precision': best_prior,
                                    }
                }
                print('nll glm', nll_map, 'elpd', nll_map, 'elpd spourious sqrt', nll_map_sqrt, 'elpd gamma', nll_map)
                with open(save_name, 'wb') as handle:
                    pickle.dump(laplace_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            test_preds_sigma = test_preds_var.squeeze().sqrt()
            pred_std_test = torch.sqrt(test_preds_sigma**2 + sigma_noise**2)

            n_train, n_val, n_test = train_preds_mu.shape[0], val_preds_mu.shape[0], test_preds_mu.shape[0]
            predictive_train = np.random.normal(train_preds_mu.detach().numpy().reshape((n_train, 1)),
                                                np.sqrt(train_preds_var.detach().numpy().reshape((n_train, 1))),
                                                size=(n_train, 200))
            predictive_val = np.random.normal(val_preds_mu.detach().numpy().reshape((n_val, 1)),
                                              np.sqrt(val_preds_var.detach().numpy().reshape((n_val, 1))),
                                              size=(n_val, 200))
            predictive_test = np.random.normal(test_preds_mu.detach().numpy().reshape((n_test, 1)),
                                               np.sqrt(test_preds_var.detach().numpy().reshape((n_test, 1))),
                                               size=(n_test, 200))

            nll_glm, elpd, elpd_sqrt = compute_metrics(predictive_train, predictive_val, predictive_test, dataset,
                                                       test_preds_mu.detach().numpy().reshape((n_test, 1)))

            print('nll glm', nll_glm, 'elpd', elpd, 'elpd spourious sqrt', elpd_sqrt)
            laplace_result_dict[f"{p}"] = {  'predictive_train': train_preds_mu.detach().numpy().reshape((n_train, 1)),
                                             'predictive_val': val_preds_mu.detach().numpy().reshape((n_val, 1)),
                                             'predictive_test': test_preds_mu.detach().numpy().reshape((n_test, 1)),
                                             'predictive_var_train': train_preds_var.detach().numpy().reshape((n_train, 1)),
                                             'predictive_var_val': val_preds_var.detach().numpy().reshape((n_val, 1)),
                                             'predictive_var_test': test_preds_var.detach().numpy().reshape((n_test, 1)),
                                             'glm_nll': nll_glm,
                                             'elpd': elpd,
                                             'elpd_spurious_sqrt': elpd_sqrt,
                                             'prior_precision': best_prior }

            with open(save_name, 'wb') as handle:
                pickle.dump(laplace_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_ignore_modules(self):
    ignore_modules = []
    for module in self.modules():
        if len([child for child in module.children()]) > 0:
            continue
        if not getattr(module, 'partial', False):
            ignore_modules.append(module)

    return ignore_modules


def change_mle_model_for_KFAC(mle_model):

    test_tensor = torch.randn(size = (1, mle_model.linear1.in_features))
    model = MAPNNLike(mle_model.linear1.in_features, 50, mle_model.output_size)
    nn.utils.vector_to_parameters(nn.utils.parameters_to_vector(mle_model.parameters()), model.parameters())
    # mle_model.non_linearity = nn.ReLU()
    assert model(test_tensor) == mle_model(test_tensor)
    return model

def run_kfac(mle_model, train_dataloader,dataset, save_name):

    layers = ['layer_one', 'layer_two', 'out']
    layer_seq = np.random.permutation(layers)
    num_params = count_parameters(mle_model)

    y_scale = torch.from_numpy(dataset.scl_Y.scale_)
    y_loc = torch.from_numpy(dataset.scl_Y.mean_)
    # Base case needed, for when find_best_prior does not converge
    best_prior = 1.0
    ph = PredictiveHelper("")
    def set_kfac_module_params(model, layers, full_grad = False):

        module_indices_kron, module_indices_jacobian = [], []
        counter = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for _ in module.parameters():
                    if name in layers:
                        module_indices_kron.append(counter)
                        module_indices_jacobian.append(counter)
                    counter += 1

        setattr(model, 'module_indices_kron', module_indices_kron)
        setattr(model, 'module_indices_jacobian', module_indices_jacobian)
        setattr(model, 'do_not_extend', True)
        return model


    laplace_result_dict = {}
    ml_model = change_mle_model_for_KFAC(mle_model)
    layers = []
    for layer in layer_seq:
        layers.append(layer)
        model = copy.deepcopy(ml_model)
        model = set_kfac_module_params(model, layers, full_grad=True)
        sigma_noise = calculate_std(model, train_dataloader, alpha=3, beta=1, beta_prior=False)

        best_prior = find_best_prior_kron(model, train_dataloader,
                                     DataLoader(UCIDataloader(dataset.X_val, dataset.y_val, ),
                                                batch_size=dataset.X_val.shape[0]),
                                     y_scale=y_scale, y_loc=y_loc, sigma_noise=sigma_noise)

        # Define and fit subnetwork LA using the specified subnetwork indices
        la = Laplace(model, 'regression',
                     subset_of_weights='all',
                     hessian_structure='kron',
                     prior_precision=torch.tensor([float(best_prior)]))

        print('Best prior was', best_prior)
        la.fit(train_dataloader)


        train_preds_mu, train_preds_var = la(torch.from_numpy(dataset.X_train).to(torch.float32))
        val_preds_mu, val_preds_var = la(torch.from_numpy(dataset.X_val).to(torch.float32))
        test_preds_mu, test_preds_var = la(torch.from_numpy(dataset.X_test).to(torch.float32))

        test_preds_sigma = test_preds_var.squeeze().sqrt()
        pred_std_test = torch.sqrt(test_preds_sigma ** 2 + sigma_noise ** 2)

        n_train, n_val, n_test = train_preds_mu.shape[0], val_preds_mu.shape[0], test_preds_mu.shape[0]
        predictive_train = np.random.normal(train_preds_mu.detach().numpy().reshape((n_train, 1)),
                                            np.sqrt(train_preds_var.detach().numpy().reshape((n_train, 1))),
                                            size=(n_train, 200))
        predictive_val = np.random.normal(val_preds_mu.detach().numpy().reshape((n_val, 1)),
                                          np.sqrt(val_preds_var.detach().numpy().reshape((n_val, 1))),
                                          size=(n_val, 200))
        predictive_test = np.random.normal(test_preds_mu.detach().numpy().reshape((n_test, 1)),
                                           np.sqrt(test_preds_var.detach().numpy().reshape((n_test, 1))),
                                           size=(n_test, 200))

        nll_glm, elpd, elpd_sqrt = compute_metrics(predictive_train, predictive_val, predictive_test, dataset,
                                                   test_preds_mu.detach().numpy().reshape((n_test, 1)))

        print('nll glm', nll_glm, 'elpd', elpd, 'elpd spourious sqrt', elpd_sqrt)
        laplace_result_dict[len(layers)] = {'predictive_train': train_preds_mu.detach().numpy().reshape((n_train, 1)),
                                       'predictive_val': val_preds_mu.detach().numpy().reshape((n_val, 1)),
                                       'predictive_test': test_preds_mu.detach().numpy().reshape((n_test, 1)),
                                       'predictive_var_train': train_preds_var.detach().numpy().reshape((n_train, 1)),
                                       'predictive_var_val': val_preds_var.detach().numpy().reshape((n_val, 1)),
                                       'predictive_var_test': test_preds_var.detach().numpy().reshape((n_test, 1)),
                                       'glm_nll': nll_glm,
                                       'elpd': elpd,
                                       'elpd_spurious_sqrt': elpd_sqrt,
                                       'prior_precision': best_prior}

        with open(save_name, 'wb') as handle:
            pickle.dump(laplace_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)




def multiple_runs(data_path,
                  dataset_class,
                  num_runs,
                  device,
                  num_epochs,
                  output_path,
                  fit_swag,
                  fit_laplace,
                  map_path,
                  dataset_path,
                  load_map=True,
                  random_mask = False,
                  kron = False,
                  largest_variance_mask = False,
                  **kwargs):
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
        if dataset_path is None:
            dataset = dataset_class(data_dir=data_path,
                                    test_split_type='random',
                                    test_size=0.1,
                                    seed=np.random.randint(0, 1000),
                                    val_fraction_of_train=0.1)
        else:
            data_run_path = os.path.join(dataset_path, f"data_laplace_run_{run}.pkl")
            dataset = pickle.load(open(data_run_path, "rb"))

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

        if not load_map:
            mle_model = train(network=mle_model, dataloader_train=train_dataloader, dataloader_val=val_dataloader,
                                 model_old = None, vi = False, device='cpu', epochs = num_epochs,
                                 save_path = os.path.join(map_path, f"run_{run}.pt"), return_best_model=True, criterion=loss_fn)
        else:
            map_run_path = os.path.join(map_path, f"run_{run}.pt")
            mle_model.load_state_dict(torch.load(map_run_path))

        if fit_laplace:
            if kron:
                save_name = os.path.join(output_path, f'results_laplace_kron_run_{run}.pkl')
                run_kfac(mle_model, train_dataloader, dataset, save_name)
            else:
                save_name = os.path.join(output_path, f'results_laplace_run_{run}.pkl')
                run_percentiles(mle_model, train_dataloader, dataset, percentages, save_name, random_mask, largest_variance_mask)

        if fit_swag:

            loss_arguments = {'loss': ll.GLLGP_loss_swag, 'prior_sigma': 1 / np.sqrt(args.prior_precision)}
            loss = loss_arguments.get('loss', MSELoss)
            loss_fn = loss(**loss_arguments)

            train_args = { 'bayes_var': False,
                           'y_scale': dataset.scl_Y.scale_,
                           'y_loc': dataset.scl_Y.mean_ ,
                           'loss': loss_fn,
                           'num_mc_samples': 200,
                           'device': args.device,
                           'epochs': args.num_epochs,
                           'save_path': args.output_path,
                           'learning_rate_sweep': np.logspace(-5, -1, 10, endpoint=True),
                           'swag_epochs': 100,
                           'calculate_std': calculate_std,
                           'random_mask': random_mask
                           }

            save_name = os.path.join(output_path, f'results_swag_run_{run}.pkl')

            swag_result_dict = train_swag(mle_model, train_dataloader, val_dataloader, test_dataloader,
                       percentages, trained_model=mle_model, train_args=train_args, dataset=dataset)
            with open(save_name, 'wb') as handle:
                pickle.dump(swag_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="boston")
    parser.add_argument('--data_path', type=str, default=os.getcwd())
    parser.add_argument('--map_path', type=str, default=os.getcwd())
    parser.add_argument('--dataset_path', type=str, default=os.getcwd())
    parser.add_argument('--num_runs', type=int, default=15)
    parser.add_argument('--size_ramping', type=ast.literal_eval, default=False)
    parser.add_argument('--get_map', type=ast.literal_eval, default=True)
    parser.add_argument('--load_map', type=ast.literal_eval, default=True)
    parser.add_argument('--fit_swag', type=ast.literal_eval, default=True)
    parser.add_argument('--fit_laplace', type=ast.literal_eval, default=True)
    parser.add_argument('--prior_precision', type=float, default=0.25)
    parser.add_argument('--random_mask', type = ast.literal_eval, default=False)
    parser.add_argument('--kron', type = ast.literal_eval, default=False)
    parser.add_argument('--largest_variance_mask', type = ast.literal_eval, default=False)

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
                      'prior_sigma': 1 / np.sqrt(args.prior_precision)}

    if args.size_ramping:
        results = make_size_ramping(args.data_path, dataset_class, args.num_runs, args.device, args.num_epochs,
                                    args.output_path, **loss_arguments)
        with open(os.path.join(args.output_path, f'results_ramping_{args.dataset}.pkl'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        multiple_runs(args.data_path, dataset_class, args.num_runs, args.device, args.num_epochs,
                      args.output_path, args.fit_swag, args.fit_laplace, args.map_path, args.dataset_path,
                      args.load_map,args.random_mask, args.kron, args.largest_variance_mask, **loss_arguments)

    print("Laplace experiments finished!")
