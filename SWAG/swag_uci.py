import copy
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from torch.utils.data import DataLoader, Dataset
from swag_temp import run_swag_partial
from VI.partial_bnn_functional import create_mask, train, create_non_parameter_mask
from MAP_baseline.MapNN import MapNN
from uci import UCIDataloader, UCIDataset, UCIBostonDataset, UCIEnergyDataset, UCIConcreteDataset, \
    UCIYachtDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import argparse
import matplotlib.pyplot as plt
from torch.distributions.gamma import Gamma
import seaborn as sns
import pandas as pd
import ast
import misc.likelihood_losses as ll
from torch.nn import MSELoss
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
        results.append(dist.log_prob(label.ravel()).item()/pred.shape[0])

    nll = -np.mean(results)
    return nll


def calculate_nll_batch(preds, labels, sigma):
    dist = MultivariateNormal(preds.ravel(), torch.eye(preds.shape[0]) * sigma)
    ll = dist.log_prob(preds.ravel()).item()/preds.shape[0]
    return -ll

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

def calculate_mse_batch(preds, labels):
    mse = torch.mean((preds - labels)**2).item()
    return mse

def calculate_nll_third(labels, mc_matrix, sigma, y_scale, y_loc):
    results = []
    for i in range(mc_matrix.shape[0]):
        res_temp = []
        for j in range(mc_matrix.shape[1]):
            dist = Normal(mc_matrix[i,j]*y_scale + y_loc, np.sqrt(sigma)*y_scale)
            res_temp.append(dist.log_prob(labels[i]*y_scale + y_loc).item())
        results.append(np.mean(res_temp))
    return np.mean(results)


def get_tau_by_conjugacy(x, alpha, beta):

    mean_x = torch.mean(x)
    n = len(x)
    dist = Gamma(alpha + n/2, beta + 1/2 * torch.sum((x - mean_x)))
    posterior_tau = torch.mean(dist.sample((1000,)))
    return posterior_tau


def calculate_nll_fourth(labels, mc_matrix, sigma, y_scale, y_loc):
    results = []
    for i in range(mc_matrix.shape[0]):
        dist = MultivariateNormal(mc_matrix[i], torch.eye(mc_matrix.shape[-1])*sigma*y_scale)
        results.append(dist.log_prob(torch.tile(labels[i]*y_scale+y_loc, (mc_matrix.shape[-1], ))).item()/mc_matrix.shape[-1])
    return np.mean(results)

def calculate_nll_fifth(labels, mc_matrix, sigma, y_scale, y_loc):

    results = []
    mc_matrix = mc_matrix.detach()
    for i in range(mc_matrix.shape[0]):
        variance = mc_matrix[i].var()
        posterior_sigma = np.sqrt(variance + sigma)
        dist = Normal(mc_matrix[i].mean()*y_scale + y_loc, posterior_sigma*y_scale)
        results.append(dist.log_prob(labels[i]*y_scale + y_loc))
    return np.mean(results)

def get_swag_residuals(model, dataloader, mask,swag_results, train_args):
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
    residuals = torch.cat(mean_preds, dim = 0) - torch.cat(all_labels, dim = 0).flatten()
    return residuals


def evaluate_swag(model,dataloader, mask, swag_results, train_args, sigma = 1):

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

        mc_overall_nll = calculate_nll_third(labels, mc_matrix_preds, sigma, y_scale.item(),y_loc.item())
        mse = calculate_mse_batch(mc_matrix_preds.mean(-1), labels.flatten())
        # test = calculate_nll_fourth(labels, mc_matrix_preds, sigma, y_scale.item(), y_loc.item())

    return np.mean(mc_overall_nll), mse


def evaluate_map(model, dataloader, sigma, y_scale, y_loc):

    for batch, label in dataloader:
        preds = model(batch)
        labels = label
        break

    nll = calculate_nll_third(labels, preds.detach(), sigma.item(), y_scale.item(), y_loc.item())
    mse = torch.mean((labels.flatten() - preds.flatten())**2)
    return nll, mse

def get_residuals(model, dataloader):
    residuals = []
    for batch, label in dataloader:
        residuals.append(model(batch)-label)

    return torch.cat(residuals, dim = 0)
def get_empirical_var(model, dataloader):

    residuals = []
    for batch, label in dataloader:
        residuals.append((model(batch)-label))

    return torch.var(torch.cat(residuals, dim = 0)).item()

def train_swag(untrained_model, dataloader, dataloader_val, dataloader_test, percentages, train_args):

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

    learning_rate_sweep = train_args['learning_rate_sweep']
    bayes_var = train_args['bayes_var']
    results_dict = {
        'nll_test': [],
        'mse_test': [],
        'best_nll_val': [],
        'according_mse_val': []
    }

    residuals = get_residuals(model_, dataloader)
    if bayes_var:
        precision = get_tau_by_conjugacy(residuals, 3, 5)
        sigma = np.sqrt(1 / precision)
    else:
        residuals = residuals.detach()
        sigma = residuals.std()

    y_scale = train_args['y_scale']
    y_loc = train_args['y_loc']
    nll, mse =  evaluate_map(model_, dataloader_val, sigma,
                             y_scale, y_loc)

    results_dict['best_nll_val'].append(nll)
    results_dict['according_mse_val'].append(mse)
    nll, mse =  evaluate_map(model_, dataloader_test, sigma,
                             y_scale, y_loc)
    results_dict['nll_test'].append(nll)
    results_dict['mse_test'].append(mse)

    model = copy.deepcopy(model_)
    for percentage in percentages:
        mask = create_non_parameter_mask(model, percentage)
        mask = mask.bool()
        nlls = []
        mses = []
        residuals = get_residuals(model, dataloader)
        precision = get_tau_by_conjugacy(residuals, 1, 1)
        sigma = np.sqrt(1 / precision)
        print("sigma without swag", sigma)
        sigmas = []
        for lr in learning_rate_sweep:
            try:
                swag_results = run_swag_partial(
                    model, dataloader, lr, n_epochs =train_args['swag_epochs'], criterion=train_args['loss'], mask=mask
                )
            except:
                breakpoint()
            residuals = get_swag_residuals(model, dataloader, mask, swag_results, train_args)
            if bayes_var:
                precision = get_tau_by_conjugacy(residuals, 3, 5)
                sigma = np.sqrt(1 / precision)
            else:
                sigma = residuals.detach().std()

            sigmas.append(sigma)
            nll, mse = evaluate_swag(model, dataloader_val, mask, swag_results, train_args, sigma = sigma)
            nlls.append(nll)
            mses.append(mse)

        print("Best Validation nll for percentage", percentage, 'was', np.min(nll),'with sigma', sigma)
        lr = learning_rate_sweep[np.argmax(nlls)]
        swag_results = run_swag_partial(
            model, dataloader, lr, n_epochs=train_args['swag_epochs'], criterion=train_args['loss'], mask=mask
        )
        nll, mse = evaluate_swag(model, dataloader_test, mask, swag_results, train_args, sigma=sigmas[np.argmax(nlls)])
        results_dict['nll_test'].append(nll)
        results_dict['mse_test'].append(mse)
        results_dict['best_nll_val'].append(np.max(nlls))
        results_dict['according_mse_val'].append(mses[np.argmax(nlls)])

    plt.figure()
    plt.plot(range(len(results_dict['nll_test'])), results_dict['nll_test'])
    plt.show(block=False)
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


        train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train//8)
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
            train_args
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

def plot_stuff(percentages, res):
    percentages = [0] + percentages
    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame()
    df['percentages'] = res.shape[0]*percentages
    df['nll'] = res.flatten() * (-1)
    plt.figure()
    sns.pointplot(errorbar=lambda x: np.percentile(x, [25, 75]),
                  data=df, x="percentages", y="nll",
                  join=False,
                  markers="d", scale=.5, errwidth=0.5, estimator=np.median)
    plt.show(block = False)
    plt.figure()
    sns.pointplot(data=df, x="percentages", y="nll",
                  join=False, errorbar=('ci', 50),
                  markers="d", scale=.5, errwidth=0.5)
    plt.show(block=False)


def plot_series(percentages, res):
    fig, ax = plt.subplots(1,1)
    perc = [0] + percentages
    rs = res - res.mean(-1)[:, None]
    rs *= -1

    runs = np.zeros_like(rs)
    for i in range(runs.shape[0]):
        runs[i] = i + 1
    df = pd.DataFrame()
    df['runs'] = runs.flatten()
    df['nll'] = rs.flatten()
    df['percentages'] = runs.shape[0] * perc
    sns.lineplot(data=df, x='percentages', y='nll', errorbar=None, ax = ax, linewidth = 2, legend=False)
    sns.lineplot(data=df, x = 'percentages', y = 'nll',
                 hue ='runs', style = 'runs', alpha = 0.4, ax = ax, legend=False,
                 palette = sns.color_palette(['black']))
    for line in ax.lines[1:]:
        line.set(linestyle = '-.')
    ax.lines[0].set_label('Mean')
    ax.legend()
    plt.show()

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

        alpha = np.random.normal(0,1, (5, 1))
        beta = np.random.normal(10, 1, (1,))
        self.X = np.random.uniform(0, 10, size = (100, 5))
        self.y = self.X@alpha + beta

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
        self.module_two = nn.Linear(10,10)
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_epochs", type=int, default=20000)
    parser.add_argument("--dataset", type=str, default="boston")
    parser.add_argument('--data_path', type=str, default=os.getcwd())
    parser.add_argument('--swag_epochs', type = int, default=50)
    parser.add_argument("--gap", type=bool, default=False)
    parser.add_argument('--num_runs', type=int, default=15)
    parser.add_argument('--bayes_var', type = ast.literal_eval, default=True)
    parser.add_argument('--prior_mu', type=int, default=0)
    parser.add_argument('--prior_precision', type=float, default=1)
    parser.add_argument('--get_map', type = ast.literal_eval, default=True)
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
                      'prior_sigma': 1/args.prior_precision}

    train_args = {
        'num_mc_samples': 100,
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