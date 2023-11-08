import copy
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from torch.utils.data import DataLoader, Dataset
from SWAG.swag_temp import run_swag_partial
from VI.partial_bnn_functional import create_mask, train, create_non_parameter_mask
from MAP_baseline.MapNN import MapNN
from uci import UCIDataloader, UCIDataset, UCIBostonDataset, UCIEnergyDataset, UCIConcreteDataset, \
    UCIYachtDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import argparse
import matplotlib.pyplot as plt
from torch.distributions.gamma import Gamma
import ast
import misc.likelihood_losses as ll
from torch.nn import MSELoss
from HMC.uci_hmc import PredictiveHelper

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
        results.append(dist.log_prob(label.ravel()).item() / pred.shape[0])

    nll = -np.mean(results)
    return nll


def calculate_nll_batch(preds, labels, sigma):
    dist = MultivariateNormal(preds.ravel(), torch.eye(preds.shape[0]) * sigma)
    ll = dist.log_prob(preds.ravel()).item() / preds.shape[0]
    return -ll


def calculate_empirical_var(preds, labels):
    results = []
    for pred, label in zip(preds, labels):
        results.append(pred - label)
    empirical_variance = torch.var(torch.cat(results, dim=0)).item()
    return empirical_variance


def calculate_mse(preds, labels):
    results = []
    for pred, label in zip(preds, labels):
        results.append((pred - label) ** 2)
    mse = torch.mean(torch.cat(results, dim=0)).item()
    return mse


def calculate_mse_batch(preds, labels):
    mse = torch.mean((preds - labels) ** 2).item()
    return mse


def calculate_nll_third(labels, mc_matrix, sigma, y_scale, y_loc):
    results = []
    for i in range(mc_matrix.shape[0]):
        res_temp = []
        for j in range(mc_matrix.shape[1]):
            dist = Normal(mc_matrix[i, j] * y_scale + y_loc, np.sqrt(sigma) * y_scale)
            res_temp.append(dist.log_prob(labels[i] * y_scale + y_loc).item())
        results.append(np.mean(res_temp))
    return np.mean(results)


def get_tau_by_conjugacy(x, alpha, beta):
    mean_x = torch.mean(x)
    n = len(x)
    dist = Gamma(alpha + n / 2, beta + 1 / 2 * torch.sum((x - mean_x)))
    posterior_tau = torch.mean(dist.sample((1000,)))
    return posterior_tau


def calculate_nll_fourth(labels, mc_matrix, sigma, y_scale, y_loc):
    results = []
    for i in range(mc_matrix.shape[0]):
        dist = MultivariateNormal(mc_matrix[i], torch.eye(mc_matrix.shape[-1]) * sigma * y_scale)
        results.append(
            dist.log_prob(torch.tile(labels[i] * y_scale + y_loc, (mc_matrix.shape[-1],))).item() / mc_matrix.shape[-1])
    return np.mean(results)


def calculate_nll_fifth(labels, mc_matrix, sigma, y_scale, y_loc):
    results = []
    mc_matrix = mc_matrix.detach()
    for i in range(mc_matrix.shape[0]):
        variance = mc_matrix[i].var()
        posterior_sigma = np.sqrt(variance + sigma)
        dist = Normal(mc_matrix[i].mean() * y_scale + y_loc, posterior_sigma * y_scale)
        results.append(dist.log_prob(labels[i] * y_scale + y_loc))
    return np.mean(results)


def get_swag_residuals(model, dataloader, mask, swag_results, train_args):
    theta_swa = swag_results["theta_swa"]
    sigma_diag = swag_results["sigma_diag"]
    D = swag_results["D"]
    K = swag_results["K"]
    y_scale = train_args['y_scale']
    y_loc = train_args['y_loc']
    model_ = copy.deepcopy(model)
    true_model_params = nn.utils.parameters_to_vector(model_.parameters()).clone()
    num_mc_samples = train_args['num_mc_samples']
    device = train_args['device']

    mean_preds = []
    all_labels = []
    for batch, labels in dataloader:
        batch, labels = batch.to(device), labels.to(device)

        mc_matrix_res = torch.zeros((labels.shape[0], num_mc_samples))
        for mc_run in range(num_mc_samples):
            z1 = torch.normal(mean=torch.zeros((theta_swa.numel())), std=1.0).to(device)
            z2 = torch.normal(mean=torch.zeros(K), std=1.0).to(device)

            theta = theta_swa + 2 ** -0.5 * (sigma_diag ** 0.5 * z1) + (2 * (K - 1)) ** -0.5 * (
                    D @ z2[:, None]).flatten()

            true_model_params[mask] = theta
            nn.utils.vector_to_parameters(true_model_params, model_.parameters())
            preds = model_(batch)
            mc_matrix_res[:, mc_run] = preds.flatten()
        mean_preds.append(mc_matrix_res.mean(-1))
        all_labels.append(labels)
    residuals = torch.cat(mean_preds, dim=0) - torch.cat(all_labels, dim=0).flatten()
    return residuals


def evaluate_swag(model, dataloader, mask, swag_results, train_args, sigma=1):
    theta_swa = swag_results["theta_swa"]
    sigma_diag = swag_results["sigma_diag"]
    D = swag_results["D"]
    K = swag_results["K"]
    y_scale = train_args['y_scale']
    y_loc = train_args['y_loc']
    model_ = copy.deepcopy(model)
    true_model_params = nn.utils.parameters_to_vector(model_.parameters()).clone()
    num_mc_samples = train_args['num_mc_samples']
    device = train_args['device']

    for batch, labels in dataloader:
        batch, labels = batch.to(device), labels.to(device)
        mc_matrix_preds = torch.zeros((batch.shape[0], num_mc_samples))

        for mc_run in range(num_mc_samples):
            z1 = torch.normal(mean=torch.zeros((theta_swa.numel())), std=1.0).to(device)
            z2 = torch.normal(mean=torch.zeros(K), std=1.0).to(device)

            theta = theta_swa + 2 ** -0.5 * (sigma_diag ** 0.5 * z1) + (2 * (K - 1)) ** -0.5 * (
                    D @ z2[:, None]).flatten()

            true_model_params[mask] = theta
            nn.utils.vector_to_parameters(true_model_params, model_.parameters())
            preds = model_(batch)
            mc_matrix_preds[:, mc_run] = preds.flatten()

        # mc_overall_nll = calculate_nll_third(labels, mc_matrix_preds, sigma, y_scale.item(), y_loc.item())
        mse = calculate_mse_batch(mc_matrix_preds.mean(-1), labels.flatten())
        nll = final_nll_calculatation(mc_matrix = mc_matrix_preds, labels = labels.flatten(),
                                      sigma = sigma,y_scale = y_scale.item(),y_loc = y_loc.item())
        # test = calculate_nll_fourth(labels, mc_matrix_preds, sigma, y_scale.item(), y_loc.item())

    return nll, mse


def final_nll_calculatation(mc_matrix, labels, sigma, y_scale, y_loc):

    fmu, fvar = mc_matrix.mean(-1), mc_matrix.std(-1)
    fvar += sigma
    labels = labels*y_scale + y_loc
    nlls = []
    for mu, var, lab in zip(fmu, fvar, labels):
        dist = Normal(mu * y_scale + y_loc, var * y_scale)
        nlls.append(
            dist.log_prob(lab.view(1, 1)).item()
        )

    return np.mean(nlls)


def get_swag_predictive(model, dataloader, mask, swag_results, train_args, sigma=1):
    theta_swa = swag_results["theta_swa"]
    sigma_diag = swag_results["sigma_diag"]
    D = swag_results["D"]
    K = swag_results["K"]
    y_scale = train_args['y_scale']
    y_loc = train_args['y_loc']
    model_ = copy.deepcopy(model)
    true_model_params = nn.utils.parameters_to_vector(model_.parameters()).clone()
    num_mc_samples = train_args['num_mc_samples']
    device = train_args['device']

    batch_preds = []
    batch_labels = []

    for batch, labels in dataloader:
        batch, labels = batch.to(device), labels.to(device)
        mc_matrix_preds = torch.zeros((batch.shape[0], num_mc_samples))

        for mc_run in range(num_mc_samples):
            z1 = torch.normal(mean=torch.zeros((theta_swa.numel())), std=1.0).to(device)
            z2 = torch.normal(mean=torch.zeros(K), std=1.0).to(device)

            theta = theta_swa + 2 ** -0.5 * (sigma_diag ** 0.5 * z1) + (2 * (K - 1)) ** -0.5 * (
                    D @ z2[:, None]).flatten()

            true_model_params[mask] = theta
            nn.utils.vector_to_parameters(true_model_params, model_.parameters())
            preds = model_(batch)
            mc_matrix_preds[:, mc_run] = preds.flatten()

        batch_preds.append(mc_matrix_preds)
        batch_labels.append(labels)

    full_preds = torch.cat(batch_preds, 0)
    full_labels = torch.cat(batch_labels, 0)
    return full_preds, full_labels


def evaluate_map(model, dataloader, sigma, y_scale, y_loc):
    for batch, label in dataloader:
        preds = model(batch)
        labels = label
        break

    nll = calculate_nll_third(labels, preds.detach(), sigma.item(), y_scale.item(), y_loc.item())
    mse = torch.mean((labels.flatten() - preds.flatten()) ** 2)
    return nll, mse


def get_residuals(model, dataloader):
    residuals = []
    for batch, label in dataloader:
        residuals.append(model(batch) - label)

    return torch.cat(residuals, dim=0)


def get_empirical_var(model, dataloader):
    residuals = []
    for batch, label in dataloader:
        residuals.append((model(batch) - label))

    return torch.var(torch.cat(residuals, dim=0)).item()


def compare_map_models(model: nn.Module, model_path, pcl_path):
    model.load_state_dict(torch.load(model_path))
    pcl = pickle.load(open(pcl_path, 'rb'))
    dataset = pcl['dataset']

    n_train, p = dataset.X_train.shape
    n_val = dataset.X_val.shape[0]
    out_dim = dataset.y_train.shape[1]
    n_test = dataset.X_test.shape[0]

    train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train // 8)
    val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)
    test_dataloader = DataLoader(UCIDataloader(dataset.X_test, dataset.y_test), batch_size=n_test)

    residuals = get_residuals(model, dataloader=train_dataloader)
    sigma = residuals.std()

    nll, mse = evaluate_map(model, test_dataloader, sigma,
                            dataset.scl_Y.scale_, dataset.scl_Y.mean_)

    return nll


def compare_for_all(model):
    path = r'C:\Users\45292\Documents\Master\MAP_models\yacht'
    model_paths = os.listdir(path)
    model_runs = []
    for p in model_paths:
        model_runs.append(int(p.split("_")[-1].split(".")[0]))

    model_paths = [os.path.join(path, p) for p in model_paths]

    res_path = r'C:\Users\45292\Documents\Master\UCI_HMC_full_tmp\UCI_HMC_VI_torch\yacht_models'
    res_paths = os.listdir(res_path)
    res_runs = []
    for p in res_paths:
        res_runs.append(int(p.split("_")[-1].split(".")[0]))

    res_paths = [os.path.join(res_path, p) for p in res_paths]
    nlls = []

    for res, rrun in zip(res_paths, res_runs):
        model_path = model_paths[model_runs.index(rrun)]
        nlls.append(compare_map_models(model, model_path, res))
    breakpoint()


def get_predictives(model, dataloaders, mask, swag_results, train_args, sigma=1):
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    pred_train, label_train = get_swag_predictive(model, train_dataloader, mask, swag_results, train_args, sigma)
    pred_test, label_test = get_swag_predictive(model, test_dataloader, mask, swag_results, train_args, sigma)
    pred_val, label_val = get_swag_predictive(model, val_dataloader, mask, swag_results, train_args, sigma)

    return {'predictions_train': pred_train.detach().numpy(), 'labels_train': label_train.detach().numpy(),
            'predictions_test': pred_test.detach().numpy(), 'labels_test': label_test.detach().numpy(),
            'predictions_val': pred_val.detach().numpy(), 'labels_val': label_val.detach().numpy()}


def train_swag(untrained_model, dataloader, dataloader_val, dataloader_test, percentages, trained_model=None,
               train_args=None):
    if trained_model is None:
        model_ = train(
            copy.deepcopy(untrained_model),
            dataloader,
            dataloader_val,
            model_old=None,
            vi=False,
            device=train_args['device'],
            epochs=train_args['epochs'],
            save_path=None,
            return_best_model=True,
            criterion=train_args['loss']
        )
    else:
        model_ = trained_model

    learning_rate_sweep = train_args['learning_rate_sweep']
    calculate_std = train_args['calculate_std']
    bayes_var = train_args['bayes_var']
    random_mask = train_args['random_mask']
    residuals = get_residuals(model_, dataloader)
    if bayes_var:
        precision = get_tau_by_conjugacy(residuals, 3, 5)
        sigma = np.sqrt(1 / precision)
    else:
        residuals = residuals.detach()
        sigma = residuals.std()

    y_scale = train_args['y_scale']
    y_loc = train_args['y_loc']
    tp = PredictiveHelper("")
    train_preds_mu = model_(torch.from_numpy(dataloader.dataset.X).to(torch.float32)).detach()
    val_preds_mu = model_(torch.from_numpy(dataloader_val.dataset.X).to(torch.float32)).detach()
    test_preds_mu = model_(torch.from_numpy(dataloader_test.dataset.X).to(torch.float32)).detach()
    sigma_noise = calculate_std(model_, dataloader, beta_prior=False)
    nll_map_sqrt = tp.calculate_nll_(test_preds_mu.numpy(), dataloader_test.dataset.y, y_scale.item(), y_loc.item(),
                                     torch.sqrt(sigma_noise))
    nll_map = tp.calculate_nll_(test_preds_mu.numpy(), dataloader_test.dataset.y, y_scale.item(), y_loc.item(),
                                sigma_noise)
    results_dict = {
        'dataset': dataloader.dataset,
        'map_results': {'map_params': model_.state_dict(),
                        'predictive_train': train_preds_mu,
                        'predictive_val': val_preds_mu,
                        'predictive_test': test_preds_mu,
                        'glm_nll': nll_map,
                        'elpd': nll_map,
                        'elpd_spurious_sqrt': nll_map_sqrt,
                        'elpd_gamma_prior': nll_map,
                        'best_lr': np.nan
                        }
    }
    model = copy.deepcopy(model_)
    for percentage in percentages:
        mask = create_non_parameter_mask(model, percentage, random_mask)
        mask = mask.bool()
        residuals = get_residuals(model, dataloader)
        precision = get_tau_by_conjugacy(residuals, 1, 1)
        sigma = np.sqrt(1 / precision)
        print("sigma without swag", sigma)
        sigmas = []
        nlls = []
        for lr in learning_rate_sweep:

            swag_results = run_swag_partial(
                model, dataloader, lr, n_epochs=train_args['swag_epochs'], criterion=train_args['loss'], mask=mask
            )
            residuals = get_swag_residuals(model, dataloader, mask, swag_results, train_args)

            if bayes_var:
                precision = get_tau_by_conjugacy(residuals, 3, 5)
                sigma = np.sqrt(1 / precision)
            else:
                sigma = residuals.detach().std()

            sigmas.append(sigma)
            nll, mse = evaluate_swag(model, dataloader_val, mask, swag_results, train_args, sigma=sigma)
            nlls.append(nll)

        print("Best Validation nll for percentage", percentage, 'was', np.min(nll), 'with sigma', sigma)
        lr = learning_rate_sweep[np.argmax(nlls)]
        swag_results = run_swag_partial(
            model, dataloader, lr, n_epochs=train_args['swag_epochs'], criterion=train_args['loss'], mask=mask
        )

        tp = PredictiveHelper("")
        predictives = get_predictives(model, (dataloader, dataloader_val, dataloader_test), mask, swag_results,
                                      train_args, sigma)
        train_, val, test = (predictives['predictions_train'],
                             predictives['predictions_val'],
                             predictives['predictions_test'])

        labels_train, labels_val, labels_test = (predictives['labels_train'],
                                                 predictives['labels_val'],
                                                 predictives['labels_test'])

        fmu, fvar = tp.glm_predictive(test, std=True)

        nll_glm = tp.glm_nll(fmu, fvar, labels_test, y_scale.item(), y_loc.item())

        residuals = tp.get_residuals(train_, labels_train, full=True)
        # print(f"MSE {np.mean(residuals.mean(1)**2)}")
        res_test = tp.get_residuals(test, labels_test, full=True)
        # print(f"MSE TEST {np.mean(res_test.mean(1)**2)}")
        sigma = tp.get_sigma(residuals.mean(1))
        # tp.plot(fmu, fvar, dataset.y_test * dataset.scl_Y.scale_.item() + dataset.scl_Y.mean_.item(),
        #         dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item())
        elpd = tp.calculate_nll_(test, labels_test, y_scale.item(), y_loc.item(),
                                 sigma ** 2)
        elpd_sqrt = tp.calculate_nll_(test, labels_test, y_scale.item(), y_loc.item(),
                                      sigma)

        results_dict[f"{percentage}"] = {'predictive_train': train_,
                                         'predictive_val': val,
                                         'predictive_test': test,
                                         'glm_nll': nll_glm,
                                         'elpd': elpd,
                                         'elpd_spurious_sqrt': elpd_sqrt,
                                         'best_lr': lr}

    return results_dict


def make_multiple_runs_swag(dataset_class, data_path, num_runs, device='cpu', gap=True, train_args=None):
    results_all = []

    loss_kwargs = train_args['loss']
    for run in range(num_runs):
        dataset = dataset_class(data_dir=data_path,
                                test_split_type="gap" if gap else "random",
                                test_size=0.1,
                                gap_column='random',
                                val_fraction_of_train=0.1,
                                seed=np.random.randint(0, 1000))

        train_args['y_scale'] = dataset.scl_Y.scale_
        train_args['y_loc'] = dataset.scl_Y.mean_

        n_train, p = dataset.X_train.shape
        n_val = dataset.X_val.shape[0]
        out_dim = dataset.y_train.shape[1]
        n_test = dataset.X_test.shape[0]
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
        # percentages = [i * 10 for i in range(1,11)]

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

        train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train // 8)
        val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)
        test_dataloader = DataLoader(UCIDataloader(dataset.X_test, dataset.y_test), batch_size=n_test)

        untrained_model = MapNN(p, 50, out_dim, "leaky_relu")

        if issubclass(loss := loss_kwargs.get('loss', MSELoss), ll.BaseMAPLossSwag):
            loss_kwargs['model'] = untrained_model
            loss_fn = loss(**loss_kwargs)
        else:
            loss_fn = loss()

        train_args['loss'] = loss_fn

        res = train_swag(
            untrained_model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            percentages,
            train_args=train_args
        )
        results['val_nll'] += res['best_nll_val']
        results['val_mse'] += res['according_mse_val']
        results['test_nll'] += res['nll_test']
        results['test_mse'] += res['mse_test']

        save_name = os.path.join(train_args['save_path'],
                                 f'results_swag_run_{run}.pkl')
        with open(save_name, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        results_all.append(results)
    return results_all


class MAPLoss(nn.Module):
    def __init__(self, prior_mu, prior_sigma):
        super(MAPLoss, self).__init__()
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.likelihood = nn.MSELoss()
        self.dist = Normal(self.prior_mu, self.prior_sigma)

    def forward(self, pred, target, w):
        out = self.likelihood(pred, target) - self.dist.log_prob(w).mean()
        return out


class DataForFun(Dataset):
    def __init__(self):
        alpha = np.random.normal(0, 1, (5, 1))
        beta = np.random.normal(10, 1, (1,))
        self.X = np.random.uniform(0, 10, size=(100, 5))
        self.y = self.X @ alpha + beta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        inp = torch.from_numpy(self.X[item]).float()
        label = torch.from_numpy(self.y[item]).float()
        return inp, label


class TestModelMulti(nn.Module):
    def __init__(self):
        super(TestModelMulti, self).__init__()

        self.module_one = nn.Linear(5, 10)
        self.module_two = nn.Linear(10, 10)
        self.activation = nn.ReLU()
        self.last_layer = nn.Linear(10, 2)

    def forward(self, x):
        out = self.module_one(x)
        out = self.activation(out)
        out = self.module_two(out)
        out = self.activation(out)
        out = self.last_layer(out)
        return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_epochs", type=int, default=20000)
    parser.add_argument("--dataset", type=str, default="boston")
    parser.add_argument('--data_path', type=str, default=os.getcwd())
    parser.add_argument('--swag_epochs', type=int, default=50)
    parser.add_argument("--gap", type=bool, default=False)
    parser.add_argument('--num_runs', type=int, default=15)
    parser.add_argument('--bayes_var', type=ast.literal_eval, default=True)
    parser.add_argument('--prior_mu', type=int, default=0)
    parser.add_argument('--prior_precision', type=float, default=1)
    parser.add_argument('--get_map', type=ast.literal_eval, default=True)
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

    loss_arguments = {'loss': ll.GLLGP_loss_swag if args.get_map else MSELoss,
                      'prior_sigma': 1 / args.prior_precision}

    train_args = {
        'num_mc_samples': 200,
        'device': args.device,
        'epochs': args.num_epochs,
        'save_path': args.output_path,
        'learning_rate_sweep': np.logspace(-1.5, -1, 2, endpoint=True),
        'swag_epochs': args.swag_epochs,
        'bayes_var': args.bayes_var,
        'loss': loss_arguments
    }

    results = []

    results = make_multiple_runs_swag(dataset_class, args.data_path, args.num_runs, args.device, args.gap, train_args)

    breakpoint()
