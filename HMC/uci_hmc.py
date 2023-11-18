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
# from Laplace.uci_laplace import calculate_std
from jax import random
numpyro.set_platform("cpu")
numpyro.set_host_device_count(8)
import argparse
import copy
from torch.distributions import Normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from misc.PredictiveHelper import PredictiveHelper


def tensor_to_jax_array(tensor):
    return jnp.array(tensor.cpu().detach().numpy())


def convert_torch_to_pyro_params(torch_params, MAP_params):
    # Torch model does not have a precision parameter
    assert len(MAP_params.keys()) - 1 == len(torch_params.keys())
    for svi_key in MAP_params.keys():
        # Hardcoded for now
        if "W1" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['linear1.weight'].detach()).T
        elif "W2" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['linear2.weight'].detach()).T
        elif "b1" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['linear1.bias'].detach()).T[:, None]
        elif "b2" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['linear2.bias'].detach()).T[:, None]
        elif "W_output" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['out.weight'].detach()).T
        elif "b_output" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['out.bias'].detach()).T[:, None]

    return MAP_params

def convert_shape_map_params(MAP_params):
    # Torch model does not have a precision parameter
    for svi_key in MAP_params.keys():
        # Hardcoded for now
        if "b1" in svi_key or "b2" in svi_key:
            MAP_params[svi_key] = MAP_params[svi_key][:, None]

    return MAP_params


def _gap_train_test_split(X, y, gap_column, test_size):
    n_data = X.shape[0]
    sorted_idxs = np.argsort(X[:, gap_column])
    train_idxs = np.concatenate(
        (
            sorted_idxs[: int(n_data * 0.5 * (1 - test_size))],
            sorted_idxs[-int(n_data * 0.5 * (1 - test_size)):],
        )
    )
    test_idxs = np.array(list(set(sorted_idxs.tolist()) - set(train_idxs.tolist())))
    X_train = X[train_idxs, :]
    X_test = X[test_idxs, :]
    y_train = y[train_idxs, :]
    y_test = y[test_idxs, :]
    return X_train, X_test, y_train, y_test


class UCIDataset:
    def __init__(
            self,
            data_dir,
            test_split_type="random",
            gap_column=0,
            test_size=0.2,
            val_fraction_of_train=0.1,
            seed=42,
            *args,
            **kwargs,
    ):
        self.data_dir = data_dir
        self.download()

        random_state = np.random.get_state()  # store current state
        X, y = self.load_from_filepath()

        np.random.seed(seed)
        if gap_column == "random":
            gap_column = np.random.randint(0, X.shape[1])

        if test_split_type == "random":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )
        elif test_split_type == "gap":
            assert gap_column in list(range(X.shape[1]))
            X_train, X_test, y_train, y_test = _gap_train_test_split(
                X, y, gap_column, test_size
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_fraction_of_train
        )

        self.scl_X = StandardScaler()
        self.scl_X.fit(X_train)
        X_train, X_val, X_test = (
            self.scl_X.transform(X_train),
            self.scl_X.transform(X_val),
            self.scl_X.transform(X_test),
        )

        self.scl_Y = StandardScaler()
        self.scl_Y.fit(y_train)
        y_train, y_val, y_test = (
            self.scl_Y.transform(y_train),
            self.scl_Y.transform(y_val),
            self.scl_Y.transform(y_test),
        )

        np.random.set_state(random_state)

        n_train, n_val, n_test = y_train.shape[0], y_val.shape[0], y_test.shape[0]
        n_total = n_train + n_val + n_test

        print(
            f"Train set: {n_train} examples, {100 * (n_train / n_total):.2f}% of all examples."
        )
        print(
            f"Val set: {n_val} examples, {100 * (n_val / n_total):.2f}% of all examples."
        )
        print(
            f"Test set: {n_test} examples, {100 * (n_test / n_total):.2f}% of all examples."
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    @property
    def file_path(self):
        return os.path.join(
            self.data_dir, "../uci_datasets", self.dataset_name, self.filename
        )

    def download(self):
        if not os.path.isfile(self.file_path):
            print(f"Downloading {self.dataset_name} Dataset")
            downloaded_file = requests.get(self.url)
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "wb") as f:
                print(f"Writing {self.dataset_name} Dataset")
                f.write(downloaded_file.content)
        else:
            print(f"{self.dataset_name} Dataset already downloaded; skipping download.")


class UCIYachtDataset(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
    filename = "yacht.data"
    dataset_name = "yacht"

    def load_from_filepath(self):
        df = pd.read_fwf(self.file_path, header=None)

        X = df.values[:-1, :-1]
        nD = X.shape[0]
        y = df.values[:-1, -1].reshape((nD, 1))

        return X, y


class UCIEnergyDataset(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    filename = "energy.xlsx"
    dataset_name = "energy"

    def load_from_filepath(self):
        df = pd.read_excel(self.file_path, engine="openpyxl")

        X = df[["X1", "X2", "X3", "X4", "X5", "X7", "X6", "X8"]].values
        nD = X.shape[0]
        y = df[["Y1"]].values.reshape((nD, 1))

        return X, y


class UCIBostonDataset(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    filename = "housing.data"
    dataset_name = "boston"

    def load_from_filepath(self):
        df = pd.read_fwf(self.file_path, header=None)

        X = df.values[:-1, :-1]
        nD = X.shape[0]
        y = df.values[:-1, -1].reshape((nD, 1))

        return X, y


class UCIConcreteDataset(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    filename = "concrete.xls"
    dataset_name = "concrete"

    def load_from_filepath(self):
        df = pd.read_excel(self.file_path, engine="xlrd")

        X = df.values[:, :-1]
        nD = X.shape[0]
        y = df.values[:, -1].reshape((nD, 1))

        return X, y


def one_d_bnn(X, y=None, prior_variance=0.1, width=50, scale=1.0):
    nB, n_features = X.shape

    W_1 = numpyro.sample(
        "W1", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((n_features, width)))
    )
    b_1 = numpyro.sample(
        "b1", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width)))
    )

    W_2 = numpyro.sample(
        "W2", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, width)))
    )
    b_2 = numpyro.sample(
        "b2", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width)))
    )

    W_output = numpyro.sample(
        "W_output", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, 1)))
    )
    b_output = numpyro.sample(
        "b_output", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((1, 1)))
    )

    z1 = X @ W_1 + b_1.reshape((1, width)).repeat(nB, axis=0)
    h1 = jax.nn.leaky_relu(z1)

    z2 = h1 @ W_2 + b_2.reshape((1, width)).repeat(nB, axis=0)
    h2 = jax.nn.leaky_relu(z2)

    output = h2 @ W_output + b_output.repeat(nB, axis=0)
    mean = numpyro.deterministic("mean", output)

    # output precision
    prec_obs = numpyro.sample(
        "prec_obs", dist.Gamma(3.0, 1.0)
    )
    # prec_obs = numpyro.sample(
    #     "prec_obs", dist.Uniform(1.,1000)
    # )

    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    with numpyro.handlers.scale(scale=scale):
        y_obs = numpyro.sample("y_obs", dist.Normal(mean, sigma_obs), obs=y)


def evaluate_samples(model, rng_key, X, y, samples, y_scale=1.0, y_loc=0.0):
    sigma_obs = (1.0 / jnp.sqrt(samples["prec_obs"])).mean()

    predictive = Predictive(model, samples)(rng_key, X=X)

    predictive_mean = (y_scale * predictive["mean"].mean(axis=0)) + y_loc
    log_likelihood = (
        dist.Normal(predictive_mean, y_scale * sigma_obs)
        .log_prob(y_loc + (y * y_scale))
        .mean()
    )
    rmse = (
                   (y_scale * (predictive["mean"].mean(axis=0).flatten() - y.flatten())) ** 2
           ).mean() ** 0.5

    return float(log_likelihood), float(rmse)


def evaluate_vi_samples(model, params, dataset, rng_key, y_scale=1.0, y_loc=0.0):
    predictive_test = Predictive(
        model=model,
        guide=autoguide.AutoNormal(model),
        params=params,
        num_samples=200,
    )(rng_key, X=dataset.X_test)

    sigma_obs = (1.0 / jnp.sqrt(params["prec_obs_auto_loc"])).mean()
    y = dataset.y_test

    # Calculate log likelihood using predictive
    predictive_mean = (y_scale * predictive_test["mean"].mean(axis=0)) + y_loc
    log_likelihood = (
        dist.Normal(predictive_mean, y_scale * sigma_obs)
        .log_prob(y_loc + (y * y_scale))
        .mean()
    )
    rmse = ((y_scale * (predictive_test["mean"].mean(axis=0).flatten() - y.flatten())) ** 2).mean() ** 0.5

    return float(log_likelihood), float(rmse)


def evaluate_samples_properly(model, rng_key, X, y, samples, y_scale=1.0, y_loc=0.0):
    sigma_obs = (1.0 / jnp.sqrt(samples["prec_obs"])).mean()

    predictive = Predictive(model, samples)(rng_key, X=X)

    predictive_mean = np.array(predictive['mean']).squeeze(-1)
    y = y.squeeze(-1)

    likelihood = calculate_ll_mc(y, predictive_mean, sigma_obs, y_scale.item(), y_loc.item())
    return likelihood


def evaluate_MAP(model, MAP_params, X, y, rng_key, y_scale=1.0, y_loc=0.0):
    predictive = Predictive(
        model=model,
        guide=autoguide.AutoDelta(model),
        params=MAP_params,
        num_samples=1,
    )(rng_key, X=X)

    sigma_obs = 1.0 / jnp.sqrt(MAP_params["prec_obs_auto_loc"])
    rmse = (((predictive["mean"][0, :] - y) ** 2).mean() ** 0.5) * y_scale
    log_likelihood = (
        dist.Normal((y_scale * predictive["mean"][0, :]) + y_loc, sigma_obs * y_scale)
        .log_prob(y_loc + (y * y_scale))
        .mean()
    )

    return float(log_likelihood), float(rmse)


def generate_mixed_bnn_by_param(
        MAP_params, sample_mask_tuple, prior_variance, scale=1.0, l_scale=1.0, use_prior=False
):
    (
        W1_sample_mask,
        W2_sample_mask,
        W_output_sample_mask,
        b1_sample_mask,
        b2_sample_mask,
        b_output_sample_mask,
    ) = sample_mask_tuple

    def mixed_bnn(X, y=None, prior_variance=prior_variance, width=50, scale=scale):
        nB, n_features = X.shape

        W_1_noise = numpyro.sample(
            "W1_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((n_features, width))),
        )
        b_1_noise = numpyro.sample(
            "b1_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones_like(MAP_params['b1_auto_loc'])),
        )
        W_2_noise = numpyro.sample(
            "W2_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, width))),
        )
        b_2_noise = numpyro.sample(
            "b2_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones_like(MAP_params['b2_auto_loc']))
        )
        W_output_noise = numpyro.sample(
            "W_output_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, 1))),
        )
        b_output_noise = numpyro.sample(
            "b_output_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((1, 1)))
        )

        W_1_map = MAP_params["W1_auto_loc"]
        b_1_map = MAP_params["b1_auto_loc"]
        W_2_map = MAP_params["W2_auto_loc"]
        b_2_map = MAP_params["b2_auto_loc"]
        W_output_map = MAP_params["W_output_auto_loc"]
        b_output_map = MAP_params["b_output_auto_loc"]

        W_1 = numpyro.deterministic(
            "W1", (W_1_map * (1 - W1_sample_mask)) + (W_1_noise * W1_sample_mask)
        )
        W_2 = numpyro.deterministic(
            "W2", (W_2_map * (1 - W2_sample_mask)) + (W_2_noise * W2_sample_mask)
        )
        W_output = numpyro.deterministic(
            "W_output",
            (W_output_map * (1 - W_output_sample_mask))
            + (W_output_noise * W_output_sample_mask),
        )

        b_1 = numpyro.deterministic(
            "b1", (b_1_map * (1 - b1_sample_mask)) + (b_1_noise * b1_sample_mask)
        )
        b_2 = numpyro.deterministic(
            "b2", (b_2_map * (1 - b2_sample_mask)) + (b_2_noise * b2_sample_mask)
        )
        b_output = numpyro.deterministic(
            "b_output",
            (b_output_map * (1 - b_output_sample_mask))
            + (b_output_noise * b_output_sample_mask),
        )

        z1 = X @ W_1 + b_1.reshape((1, width)).repeat(nB, axis=0)
        h1 = jax.nn.leaky_relu(z1)

        z2 = h1 @ W_2 + b_2.reshape((1, width)).repeat(nB, axis=0)
        h2 = jax.nn.leaky_relu(z2)

        output = h2 @ W_output + b_output.repeat(nB, axis=0)
        mean = numpyro.deterministic("mean", output)

        # output precision
        if use_prior:
            prec_obs = numpyro.sample(
                "prec_obs", dist.Gamma(3.0, 1.0)
            )  # MAP outperforms full BNN, even if we freeze the prior precision. That's interesting here, I think.
            sigma_obs = 1.0 / jnp.sqrt(prec_obs)
            with numpyro.handlers.scale(scale=scale):
                y_obs = numpyro.sample("y_obs", dist.Normal(mean, sigma_obs), obs=y)
        else:
            with numpyro.handlers.scale(scale=scale):
                y_obs = numpyro.sample("y_obs", dist.Normal(mean, l_scale), obs=y)

    return mixed_bnn


def generate_node_based_bnn(MAP_params, sample_mask_tuple, prior_variance, scale=1.0, l_scale = 1, use_prior=True, is_svi_map = False):

    (
        W1_sample_mask,
        W2_sample_mask,
        W_output_sample_mask,
        b1_sample_mask,
        b2_sample_mask,
        b_output_sample_mask
    ) = sample_mask_tuple

    def mixed_bnn(X, y=None, prior_variance=prior_variance, width=50, scale=scale):
        nB, n_features = X.shape

        W_1_node_noise = numpyro.sample(
            "W1_node_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((n_features, 1))),
        )
        b_1_node_noise = numpyro.sample(
            "b1_node_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones_like(MAP_params['b1_auto_loc'])),
        )
        W_2_node_noise = numpyro.sample(
            "W2_node_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, 1))),
        )
        b_2_node_noise = numpyro.sample(
            "b2_node_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones_like(MAP_params['b2_auto_loc']))
        )
        W_output_node_noise = numpyro.sample(
            "W_output_node_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, 1))),
        )
        b_output_node_noise = numpyro.sample(
            "b_output_node_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((1, 1)))
        )

        W_1_map = MAP_params["W1_auto_loc"]
        b_1_map = MAP_params["b1_auto_loc"]
        W_2_map = MAP_params["W2_auto_loc"]
        b_2_map = MAP_params["b2_auto_loc"]
        W_output_map = MAP_params["W_output_auto_loc"]
        b_output_map = MAP_params["b_output_auto_loc"]

        W_1_node = numpyro.deterministic(
            "W_1_node", (jnp.ones_like(W1_sample_mask) * (1 - W1_sample_mask)) + (W_1_node_noise * W1_sample_mask)
        )

        W_2_node = numpyro.deterministic(
            "W_2_node", (jnp.ones_like(W2_sample_mask) * (1 - W2_sample_mask)) + (W_2_node_noise * W2_sample_mask)
        )
        W_output_node = numpyro.deterministic(
            "W_output_node",
            (jnp.ones_like(W_output_sample_mask) * (1 - W_output_sample_mask))
            + (W_output_node_noise * W_output_sample_mask),
        )

        b_1_node = numpyro.deterministic(
            "b_1_node", (jnp.ones_like(b1_sample_mask) * (1 - b1_sample_mask)) + (b_1_node_noise * b1_sample_mask)
        )

        b_2_node = numpyro.deterministic(
            "b_2_node", (jnp.ones_like(b2_sample_mask) * (1 - b2_sample_mask)) + (b_2_node_noise * b2_sample_mask)
        )

        b_output_node = numpyro.deterministic(
            "b_output_node",
            (jnp.ones_like(b_output_sample_mask) * (1 - b_output_sample_mask))
            + (b_output_node_noise * b_output_sample_mask),
        )

        z1 = ((X * W_1_node.reshape(1, -1).repeat(nB, axis=0)) @ W_1_map) \
             + b_1_map.reshape((1, width)).repeat(nB, axis=0)

        z1 = z1 * b_1_node.reshape((1, width)).repeat(nB, axis=0)
        # z1 = ((X @ W_1_map) * W_1_node + b_1_map.reshape((1, width)).repeat(nB, axis=0))*b_1_node
        h1 = jax.nn.leaky_relu(z1)
        z2 = ((h1 * W_2_node.reshape(1, -1).repeat(nB, axis=0)) @ W_2_map) + \
             b_2_map.reshape((1, width)).repeat(nB, axis=0)

        z2 = z2 * b_2_node.reshape((1, width)).repeat(nB, axis=0)
        # z2 = ((h1 @ W_2_map) * W_2_node + b_2_map.reshape((1, width)).repeat(nB, axis=0)) * b_2_node
        h2 = jax.nn.leaky_relu(z2)
        output = ((h2 * W_output_node.reshape(1, -1).repeat(nB, axis=0)) @ W_output_map) \
                 + b_output_map.repeat(nB, axis=0)

        output = output * b_output_node.repeat(nB, axis=0)
        # output = ((h2 @ W_output_map)*W_output_node + b_output_map.repeat(nB, axis=0)) * b_output_node

        mean = numpyro.deterministic("mean", output)

        if use_prior:
            prec_obs = numpyro.sample(
                "prec_obs", dist.Gamma(3.0, 1.0)
            )  # MAP outperforms full BNN, even if we freeze the prior precision. That's interesting here, I think.
            sigma_obs = 1.0 / jnp.sqrt(prec_obs)
            with numpyro.handlers.scale(scale=scale):
                y_obs = numpyro.sample("y_obs", dist.Normal(mean, sigma_obs), obs=y)
        else:
            with numpyro.handlers.scale(scale=scale):
                y_obs = numpyro.sample("y_obs", dist.Normal(mean, l_scale), obs=y)

    return mixed_bnn


def generate_additive_node_based_bnn(MAP_params, sample_mask_tuple, prior_variance, scale=1.0, l_scale=1,
                                     use_prior = True):
    (
        W1_sample_mask,
        W2_sample_mask,
        W_output_sample_mask,
        b1_sample_mask,
        b2_sample_mask,
        b_output_sample_mask
    ) = sample_mask_tuple

    def mixed_bnn(X, y=None, prior_variance=prior_variance, width=50, scale=scale):
        nB, n_features = X.shape

        s1_node_noise = numpyro.sample(
            's1_node_noise',
            dist.Normal(0, (prior_variance ** 0.5)* jnp.ones((n_features, 1)))
        )
        r1_node_noise = numpyro.sample(
            'r1_node_noise',
            dist.Normal(0, (prior_variance ** 0.5)* jnp.ones((width, 1)))
        )

        s2_node_noise = numpyro.sample(
            's2_node_noise',
            dist.Normal(0, (prior_variance ** 0.5)* jnp.ones((width, 1)))
        )
        r2_node_noise = numpyro.sample(
            'r2_node_noise',
            dist.Normal(0, (prior_variance ** 0.5)* jnp.ones((width, 1)))
        )

        s_output_node_noise = numpyro.sample(
            's_output_node_noise',
            dist.Normal(0, (prior_variance ** 0.5)* jnp.ones((width, 1)))
        )
        r_output_node_noise = numpyro.sample(
            'r_output_node_noise',
            dist.Normal(0, (prior_variance ** 0.5)* jnp.ones((1, 1)))
        )

        W_1_map = MAP_params["W1_auto_loc"]
        b_1_map = MAP_params["b1_auto_loc"]
        W_2_map = MAP_params["W2_auto_loc"]
        b_2_map = MAP_params["b2_auto_loc"]
        W_output_map = MAP_params["W_output_auto_loc"]
        b_output_map = MAP_params["b_output_auto_loc"]

        W_1_node = numpyro.deterministic(
            "W_1_node", ((r1_node_noise @ s1_node_noise.T).T * W1_sample_mask)
        )

        W_2_node = numpyro.deterministic(
            "W_2_node", ((r2_node_noise @ s2_node_noise.T).T * W2_sample_mask)
        )
        W_output_node = numpyro.deterministic(
            "W_output_node",
            ((r_output_node_noise @ s_output_node_noise.T).T * W_output_sample_mask),
        )

        z1 = (X @ (W_1_map + W_1_node)) + b_1_map.reshape((1, width)).repeat(nB, axis=0)

        # z1 = ((X @ W_1_map) * W_1_node + b_1_map.reshape((1, width)).repeat(nB, axis=0))*b_1_node
        h1 = jax.nn.leaky_relu(z1)
        z2 = (h1 @ (W_2_node + W_2_map)) + b_2_map.reshape((1, width)).repeat(nB, axis=0)

        # z2 = ((h1 @ W_2_map) * W_2_node + b_2_map.reshape((1, width)).repeat(nB, axis=0)) * b_2_node
        h2 = jax.nn.leaky_relu(z2)
        output = (h2 @ (W_output_node + W_output_map)) + b_output_map.repeat(nB, axis=0)

        # output = ((h2 @ W_output_map)*W_output_node + b_output_map.repeat(nB, axis=0)) * b_output_node

        mean = numpyro.deterministic("mean", output)

        if use_prior:
            prec_obs = numpyro.sample(
                "prec_obs", dist.Gamma(3.0, 1.0)
            )  # MAP outperforms full BNN, even if we freeze the prior precision. That's interesting here, I think.
            sigma_obs = 1.0 / jnp.sqrt(prec_obs)
            with numpyro.handlers.scale(scale=scale):
                y_obs = numpyro.sample("y_obs", dist.Normal(mean, sigma_obs), obs=y)
        else:
            with numpyro.handlers.scale(scale=scale):
                y_obs = numpyro.sample("y_obs", dist.Normal(mean, l_scale), obs=y)

    return mixed_bnn


def calculate_ll_mc(labels, mc_matrix, sigma, y_scale, y_loc):
    results = []
    for i in range(mc_matrix.shape[1]):
        res_temp = []
        for j in range(mc_matrix.shape[0]):
            dist = Normal(mc_matrix[j, i].item() * y_scale + y_loc, np.sqrt(sigma) * y_scale)
            res_temp.append(dist.log_prob(torch.tensor([labels[i].item()]) * y_scale + y_loc).item())
        results.append(np.mean(res_temp))
    return np.mean(results)


def create_predictives(model, params, dataset, bnn, num_mc_samples=200, delta=False):
    if delta:
        guide = lambda: autoguide.AutoDelta(bnn)
    else:
        guide = lambda: autoguide.AutoNormal(bnn)

    predictive_train = Predictive(
        model=model,
        guide=guide(),
        params=params,
        num_samples=num_mc_samples,
    )(rng_key, X=dataset.X_train)

    predictive_val = Predictive(
        model=model,
        guide=guide(),
        params=params,
        num_samples=num_mc_samples,
    )(rng_key, X=dataset.X_val)

    predictive_test = Predictive(
        model=model,
        guide=guide(),
        params=params,
        num_samples=num_mc_samples,
    )(rng_key, X=dataset.X_test)

    return predictive_train, predictive_val, predictive_test


def calculate_ll_ours(model, params, dataset, bnn, num_mc_samples=200, delta=False):
    predictive_train, predictive_val, predictive_test = create_predictives(model, params, dataset, bnn, num_mc_samples,
                                                                           delta)
    y_scale = dataset.scl_Y.scale_
    y_loc = dataset.scl_Y.mean_
    ytrain, yval, ytest = dataset.y_train.squeeze(), dataset.y_val.squeeze(), dataset.y_test.squeeze()

    if num_mc_samples == 1:
        ptrain = np.asarray(predictive_train['mean'].squeeze(0))
        ptest = np.asarray(predictive_test['mean'].squeeze(0))
        pval = np.asarray(predictive_val['mean'].squeeze(0))
        sigma = np.std(ptrain - ytrain)
    else:
        ptrain = np.asarray(predictive_train['mean'].squeeze(-1))
        ptest = np.asarray(predictive_test['mean'].squeeze(-1))
        pval = np.asarray(predictive_val['mean'].squeeze(-1))
        sigma = np.std(ptrain - ytrain)

    test_ll = calculate_ll_mc(ytest, ptest, sigma, y_scale.item(), y_loc.item())
    val_ll = calculate_ll_mc(yval, pval, sigma, y_scale.item(), y_loc.item())

    return test_ll, val_ll


def create_sample_mask_random(percentile, MAP_params, vals):
    keys = [
        "W1_auto_loc",
        "W2_auto_loc",
        "W_output_auto_loc",
        "b1_auto_loc",
        "b2_auto_loc",
        "b_output_auto_loc",
    ]

    # masks = [np.zeros((MAP_params[key].shape[0], )) for key in keys]
    masks_rng = vals
    param_abs_values = np.abs(np.concatenate(masks_rng))
    val = np.percentile(param_abs_values, 100 - percentile)

    W1_sample_mask = np.abs(masks_rng[0]) >= val
    W2_sample_mask = np.abs(masks_rng[1]) >= val
    W_output_sample_mask = np.abs(masks_rng[2]) >= val
    b1_sample_mask = np.abs(masks_rng[3]) >= val
    b2_sample_mask = np.abs(masks_rng[4]) >= val
    b_output_sample_mask = np.abs(masks_rng[5]) >= val

    sample_mask_tuple = (
        W1_sample_mask[:, None],
        W2_sample_mask[:, None],
        W_output_sample_mask[:, None],
        b1_sample_mask[:, None],
        b2_sample_mask[:, None],
        b_output_sample_mask[:, None],
    )

    return sample_mask_tuple

def create_sample_mask_largest_inf_norm(percentile, MAP_params):
    keys = [
        "W1_auto_loc",
        "W2_auto_loc",
        "W_output_auto_loc",
        "b1_auto_loc",
        "b2_auto_loc",
        "b_output_auto_loc",
    ]

    inf_norms = []
    for key in keys:
        inf_norms.append(np.abs(MAP_params[key]).max(-1))

    all_values = np.concatenate(inf_norms)
    val = np.percentile(all_values, 100 - percentile)

    W1_sample_mask = inf_norms[0] >= val
    W2_sample_mask = inf_norms[1] >= val
    W_output_sample_mask = inf_norms[2] >= val
    b1_sample_mask = inf_norms[3] >= val
    b2_sample_mask = inf_norms[4] >= val
    b_output_sample_mask = inf_norms[5] >= val

    sample_mask_tuple = (
        W1_sample_mask.reshape(-1, 1),
        W2_sample_mask.reshape(-1, 1),
        W_output_sample_mask.reshape(-1, 1),
        b1_sample_mask.reshape(-1, 1),
        b2_sample_mask.reshape(-1, 1),
        b_output_sample_mask.reshape(-1, 1),
    )

    return sample_mask_tuple

def create_sample_mask_largest_abs_values(percentile, MAP_params, random_mask = False):
    keys = [
        "W1_auto_loc",
        "W2_auto_loc",
        "W_output_auto_loc",
        "b1_auto_loc",
        "b2_auto_loc",
        "b_output_auto_loc",
    ]


    if random_mask:
        new_map_params = {key: np.random.normal(size = MAP_params[key].shape) for key in keys}
    else:
        new_map_params = copy.deepcopy(MAP_params)


    all_values = np.concatenate([new_map_params[key].ravel() for key in keys])
    param_abs_values = np.abs(all_values)
    val = np.percentile(param_abs_values, 100 - percentile)

    W1_sample_mask = np.abs(new_map_params["W1_auto_loc"]) >= val
    W2_sample_mask = np.abs(new_map_params["W2_auto_loc"]) >= val
    W_output_sample_mask = np.abs(new_map_params["W_output_auto_loc"]) >= val
    b1_sample_mask = np.abs(new_map_params["b1_auto_loc"]) >= val
    b2_sample_mask = np.abs(new_map_params["b2_auto_loc"]) >= val
    b_output_sample_mask = np.abs(new_map_params["b_output_auto_loc"]) >= val

    sample_mask_tuple = (
        W1_sample_mask,
        W2_sample_mask,
        W_output_sample_mask,
        b1_sample_mask,
        b2_sample_mask,
        b_output_sample_mask,
    )

    return sample_mask_tuple


def compute_metrics(predictive_train, predictive_val, predictive_test, dataset, sigma_obs):
    rr = {'predictive_train': predictive_train['mean'], 'predictive_val': predictive_val['mean'],
          'predictive_test': predictive_test['mean']}
    tp = PredictiveHelper("")
    train, val, test = tp.convert_to_proper_format(rr)
    fmu, fvar = tp.glm_predictive(test, std=True)
    try:
        nll_glm = tp.glm_nll(fmu, fvar, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item())
    except:
        nll_glm = np.inf
    residuals = tp.get_residuals(train, dataset.y_train, full=True)
    # print(f"MSE {np.mean(residuals.mean(1)**2)}")
    res_test = tp.get_residuals(test, dataset.y_test, full=True)
    # print(f"MSE TEST {np.mean(res_test.mean(1)**2)}")
    sigma = tp.get_sigma(residuals.mean(1))
    # tp.plot(fmu, fvar, dataset.y_test * dataset.scl_Y.scale_.item() + dataset.scl_Y.mean_.item(),
    #         dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item())
    elpd = tp.calculate_nll_(test, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(), sigma ** 2)
    elpd_sqrt = tp.calculate_nll_(test, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(),
                                  sigma)

    elpd_gamma_prior = tp.calculate_nll_(test, dataset.y_test, dataset.scl_Y.scale_.item(),
                                         dataset.scl_Y.mean_.item(),
                                         sigma_obs)
    return nll_glm, elpd, elpd_sqrt, elpd_gamma_prior



def run_for_percentile(
        dataset,
        percentile,
        MAP_params,
        prior_variance=0.8,
        prior_variance_scaled=True,
        scale=1.0,
        random_mask = False
):
    sample_mask_tuple = create_sample_mask_largest_abs_values(percentile, MAP_params, random_mask)
    prior_variance_used = (
        prior_variance
        if not prior_variance_scaled
        else prior_variance * (100 / percentile)
    )

    mixed_bnn = generate_mixed_bnn_by_param(
        MAP_params,
        create_sample_mask_largest_abs_values(percentile, MAP_params, random_mask),
        prior_variance_used,
        scale=scale,
        use_prior=True
    )

    nuts_kernel = NUTS(mixed_bnn, max_tree_depth=15)
    mcmc = MCMC(nuts_kernel, num_warmup=325, num_samples=75, num_chains=8)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, dataset.X_train, dataset.y_train)

    samples = mcmc.get_samples()
    predictive_train = Predictive(mixed_bnn, samples)(rng_key, X=dataset.X_train)
    predictive_val = Predictive(mixed_bnn, samples)(rng_key, X=dataset.X_val)
    predictive_test = Predictive(mixed_bnn, samples)(rng_key, X=dataset.X_test)

    nll_glm, elpd, elpd_sqrt, elpd_gamma_prior = compute_metrics(predictive_train, predictive_val, predictive_test,
                                                                 dataset, sigma_obs = (1.0 / jnp.sqrt(samples["prec_obs"])).mean())

    test_ll_homoscedastic, _ = evaluate_samples(mixed_bnn, rng_key, dataset.X_test, dataset.y_test,
                                         mcmc.get_samples(), y_scale=dataset.scl_Y.scale_,
                                         y_loc=dataset.scl_Y.mean_)
    results = { 'predictive_train': predictive_train["mean"],
                'predictive_val': predictive_val["mean"],
                'predictive_test': predictive_test["mean"],
                'glm_nll': nll_glm,
                'elpd': elpd,
                'elpd_spurious_sqrt': elpd_sqrt,
                'elpd_gamma_prior': elpd_gamma_prior,
                'test_ll_homoscedastic': test_ll_homoscedastic
                }
    return results


def make_vi_run(run, dataset_, prior_variance, scale, results_dict, save_path, num_epochs, MAP_params_,
                node_based=True, add_node_based = False,l_scale=1.0, is_svi_map=False, inf_norm_mask = False,
                random_mask = False, only_full = False):
    rng_key = random.PRNGKey(1)
    optimizer = numpyro.optim.Adam(0.01)
    percentiles = [1, 2, 5, 8, 14, 23, 37, 61, 100]
    if only_full:
        percentiles = [100 for _ in range(len(percentiles))]

    if node_based:
        keys = ["W1_auto_loc", "W2_auto_loc", "W_output_auto_loc", "b1_auto_loc", "b2_auto_loc", "b_output_auto_loc"]
        mask_values = [np.random.normal(0, 1, size=(MAP_params_[key].shape[0],)) for key in keys]
        save_name = os.path.join(save_path, f"results_vi_node_run_{run}.pkl")
    elif add_node_based:
        save_name = os.path.join(save_path, f"results_vi_node_add_run_{run}.pkl")
    else:
        save_name = os.path.join(save_path, f"results_vi_run_{run}.pkl")
    counter = 0
    for percentile in percentiles:
        prior_variance = 100 - percentile + 1
        print("Running for percentile: ", percentile, "%")
        sample_mask_tuple = create_sample_mask_largest_abs_values(percentile, MAP_params_)
        if node_based:
            if inf_norm_mask:
                if random_mask:
                    raise ValueError("You have selected both random mask, and inf_norm_mask, both "
                                "cannot be at the same time")

                sample_mask_tuple = create_sample_mask_largest_inf_norm(percentile, MAP_params_)
            else:
                sample_mask_tuple = create_sample_mask_random(percentile, MAP_params_, mask_values)
        else:
            sample_mask_tuple = create_sample_mask_largest_abs_values(percentile, MAP_params_, random_mask)

        if node_based:
            mixed_bnn = generate_node_based_bnn(
                MAP_params_,
                sample_mask_tuple,
                prior_variance,
                scale=scale,
                l_scale=l_scale,
                use_prior=True,
                is_svi_map=is_svi_map)

            model = lambda X, y=None: generate_node_based_bnn(
                MAP_params_, sample_mask_tuple, prior_variance, scale, use_prior=True
            )(X, y)
        elif add_node_based:
            mixed_bnn = generate_additive_node_based_bnn(
                MAP_params_,
                sample_mask_tuple,
                prior_variance,
                scale=scale,
                l_scale=l_scale,
                use_prior=True)

            model = lambda X, y=None: generate_additive_node_based_bnn(
                MAP_params_, sample_mask_tuple, prior_variance, scale, use_prior=True
            )(X, y)
        else:
            mixed_bnn = generate_mixed_bnn_by_param(
                MAP_params_,
                sample_mask_tuple,
                prior_variance,
                scale=scale,
                l_scale=l_scale,
                use_prior=True
            )
            model = lambda X, y=None: generate_mixed_bnn_by_param(
                MAP_params_, sample_mask_tuple, prior_variance, scale, use_prior=True
            )(X, y)

        svi = SVI(model, autoguide.AutoNormal(mixed_bnn), optimizer, Trace_ELBO())
        svi_results = svi.run(rng_key, num_epochs, X=dataset_.X_train, y=dataset_.y_train)

        # Evaluate the model
        predictive_train, predictive_val, predictive_test = create_predictives(model, svi_results.params,
                                                                               dataset_,
                                                                               mixed_bnn, num_mc_samples=200,
                                                                               delta=False)

        nll_glm, elpd, elpd_sqrt, elpd_gamma_prior = compute_metrics(predictive_train, predictive_val,
                                                                     predictive_test, dataset_,
                                                                     sigma_obs=(svi_results.params['prec_obs_auto_scale'] ** (-1)).item())

        print('nll glm', nll_glm, 'elpd', elpd, 'elpd spourious sqrt', elpd_sqrt, 'elpd gamma',
              elpd_gamma_prior)
        results_dict[f"{percentile}" if not only_full else f"{percentile}_{counter}"] = \
            {'predictive_train': predictive_train["mean"],
                                         'predictive_val': predictive_val["mean"],
                                         'predictive_test': predictive_test["mean"],
                                         'glm_nll': nll_glm,
                                         'elpd': elpd,
                                         'elpd_spurious_sqrt': elpd_sqrt,
                                         'elpd_gamma_prior': elpd_gamma_prior}
        counter += 1

        with open(save_name, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_MAP_solution(mle_model, dataset, num_epochs):
    loss_arguments = {'loss': GLLGP_loss_swag, 'prior_sigma': 1}
    if issubclass(loss := loss_arguments.get('loss', MSELoss), BaseMAPLossSwag):
        loss_arguments['model'] = mle_model
        loss_fn = loss(**loss_arguments)

    train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train // 8)
    val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)

    mle_model = train(network=mle_model, dataloader_train=train_dataloader, dataloader_val=val_dataloader,
                      model_old=None, vi=False, device='cpu', epochs=num_epochs, return_best_model=True,
                      criterion=loss_fn)
    return mle_model


def make_hmc_run(run, dataset, scale_prior, prior_variance, save_path, likelihood_scale, percentiles, results_dict,
                 MAP_params, random_mask = False):
    for percentile in percentiles:
        # If update runs are done
        if str(percentile) not in results_dict.keys():
            print(f"Running for {percentile} of weights sampled scaled, by maximum absolute value")
            results_dict[f"{percentile}"] = run_for_percentile(
                dataset,
                percentile,
                MAP_params,
                prior_variance_scaled=scale_prior,
                prior_variance=prior_variance,
                scale=likelihood_scale,
                random_mask = random_mask
            )
            with open(os.path.join(save_path, f"results_hmc_run_{run}.pkl"), "wb") as handle:
                pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved results for ", percentile, "%")


def predictive_(model, params, X):
    predictive = Predictive(
        model=model,
        guide=autoguide.AutoDelta(model),
        params=params,
        num_samples=1,
    )(rng_key, X=X)
    return predictive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dataset", type=str, default="yacht")
    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--data_path", type=str, default=os.getcwd())
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--scale_prior",  type=ast.literal_eval, default=True)
    parser.add_argument("--prior_variance", type=float, default=4.0) #0.1 is good for yacht, 4.0 for other datasets
    parser.add_argument("--likelihood_scale", type=float, default=1.0) #6.0 is good for yacht, 1.0   for other datasets
    parser.add_argument('--vi', type=ast.literal_eval, default=True)
    parser.add_argument('--node_based', type=ast.literal_eval, default=True)
    parser.add_argument('--hmc', type=ast.literal_eval, default=True)
    parser.add_argument('--l_var', type=float, default=1.0)
    parser.add_argument('--node_based_add', type=ast.literal_eval, default=True)
    parser.add_argument('--inf_norm_mask', type=ast.literal_eval, default=False)
    parser.add_argument('--random_mask', type=ast.literal_eval, default=False)
    parser.add_argument('--only_full', type=ast.literal_eval, default = False)
    args = parser.parse_args()

    if args.inf_norm_mask and not args.node_based:
        UserWarning("inf norm mask only implemented for node based multiplicative, it has no effect in current run")

    if args.inf_norm_mask and args.random_mask:
        raise ValueError("You selected both random and inf_norm_mask, both cannot be set to True")

    if args.dataset == "yacht":
        dataset_class = UCIYachtDataset
    elif args.dataset == "energy":
        dataset_class = UCIEnergyDataset
    elif args.dataset == "boston":
        dataset_class = UCIBostonDataset

    percentiles = [1, 2, 5, 8, 14, 23, 37, 61, 100]

    if args.dataset_path is None:
        rand_seed = np.random.randint(0, 10000)
        dataset = dataset_class(
            args.data_path,
            seed=rand_seed,
            test_split_type="random",
            test_size=0.1,
            val_fraction_of_train=0.1,
        )
    else:
        dataset = pickle.load(open(args.dataset_path, "rb"))

    # Allowing for both types of MAP models
    is_svi_map = args.map_path is None
    rng_key = random.PRNGKey(1)
    optimizer = numpyro.optim.Adam(0.01)
    model = lambda X, y=None: one_d_bnn(X, y, prior_variance=args.prior_variance)
    new_map = False
    # Allows for re-runs

    new_map = True

    # Setup the MAP model
    svi = SVI(model, autoguide.AutoDelta(one_d_bnn), optimizer, Trace_ELBO())
    svi_results = svi.run(rng_key, 2, X=dataset.X_train, y=dataset.y_train)

    MAP_params = svi_results.params

    if is_svi_map:
        svi_results = svi.run(rng_key, 20000, X=dataset.X_train, y=dataset.y_train)
        MAP_params = svi_results.params
        MAP_params = convert_shape_map_params(MAP_params)
        test_ll_homoscedastic, _ = evaluate_MAP(model, MAP_params, dataset.X_test, dataset.y_test,
                                                rng_key, y_scale=dataset.scl_Y.scale_, y_loc=dataset.scl_Y.mean_)
    else:
        # Setup pytorch MAP
        n_train, p = dataset.X_train.shape
        n_val = dataset.X_val.shape[0]
        out_dim = dataset.y_train.shape[1]
        mle_model = MapNN(input_size=p, width=50, output_size=out_dim, non_linearity="leaky_relu")
        mle_model.load_state_dict(torch.load(args.map_path))
        mle_state_dict = mle_model.state_dict()
        MAP_params = convert_torch_to_pyro_params(mle_state_dict, MAP_params)
        test_ll_homoscedastic = 0.0


    predictive_train, predictive_val, predictive_test = create_predictives(model, MAP_params, dataset, one_d_bnn,
                                                                           num_mc_samples=1, delta=True)
    rr = {'predictive_train': predictive_train['mean'], 'predictive_val': predictive_val['mean'],
          'predictive_test': predictive_test['mean']}
    tp = PredictiveHelper("")

    train, val, test = tp.convert_to_proper_format(rr)
    residuals = tp.get_residuals(train, dataset.y_train, full=True)
    print(f"MSE {np.mean(residuals.mean(1) ** 2)}")
    res_test = tp.get_residuals(test, dataset.y_test, full=True)
    print("MSE Test", np.mean(res_test.mean(1) ** 2))
    sigma = tp.get_sigma(residuals.mean(1))
    print("SIGMA", sigma)
    fvar = np.ones_like(test) * np.sqrt(sigma)

    nll_one = tp.calculate_nll_(test, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(),
                                sigma ** 2)
    nll_two = tp.calculate_nll_(test, dataset.y_test, dataset.scl_Y.scale_.item(), dataset.scl_Y.mean_.item(),
                                sigma)
    print("Nll ", nll_one)
    print("Nll spurious sqrt", nll_two)

    hmc_result_dict = {
        'dataset': dataset,
        'map_results': {'map_params': MAP_params,
                        'predictive_train': predictive_train["mean"],
                        'predictive_val': predictive_val["mean"],
                        'predictive_test': predictive_test["mean"],
                        'glm_nll': nll_one,
                        'elpd': nll_one,
                        'elpd_spurious_sqrt': nll_two,
                        'elpd_gamma_prior': nll_one,
                        'test_ll_homoscedastic': test_ll_homoscedastic
                        }
    }

    results_dict_init = {
                        'dataset': dataset,
                        'map_results':  {'map_params': MAP_params,
                                         'predictive_train': predictive_train["mean"],
                                         'predictive_val': predictive_val["mean"],
                                         'predictive_test': predictive_test["mean"],
                                         'glm_nll': nll_one,
                                         'elpd': nll_one,
                                         'elpd_spurious_sqrt': nll_two,
                                         'elpd_gamma_prior': nll_one
                                         }
                     }
    if args.node_based_add:
        MAP_params = results_dict_init['map_results']['map_params']
        print("Running node based rank 1 VI")
        make_vi_run(run=args.run, dataset_=dataset, prior_variance=args.prior_variance, scale=args.likelihood_scale, results_dict=results_dict_init,
                    MAP_params_=MAP_params, save_path=args.output_path, num_epochs=args.num_epochs,
                    node_based=False, add_node_based=True, l_scale=args.l_var, is_svi_map=is_svi_map,
                    random_mask=args.random_mask, only_full=args.only_full)

    if args.node_based:
        MAP_params = results_dict_init['map_results']['map_params']
        print("Running node based VI")
        make_vi_run(run=args.run, dataset_=dataset, prior_variance=args.prior_variance, scale=args.likelihood_scale, results_dict=results_dict_init,
                    MAP_params_=MAP_params, save_path=args.output_path, num_epochs=args.num_epochs, node_based=True, l_scale=args.l_var, is_svi_map=is_svi_map,
                    inf_norm_mask=args.inf_norm_mask, random_mask=args.random_mask, only_full=args.only_full)

    if args.vi:
        MAP_params = results_dict_init['map_results']['map_params']
        print("Running VI")
        make_vi_run(run=args.run, dataset_=dataset, prior_variance=args.prior_variance, scale=args.likelihood_scale,
                    results_dict=results_dict_init,
                    MAP_params_=MAP_params, save_path=args.output_path, num_epochs=args.num_epochs, node_based=False,
                    l_scale=args.l_var,
                    random_mask=args.random_mask, only_full=args.only_full)

    if args.hmc:
        MAP_params = hmc_result_dict['map_results']['map_params']
        make_hmc_run(run=args.run, dataset=dataset, scale_prior=args.scale_prior,
                     prior_variance=args.prior_variance,
                     save_path=args.output_path, likelihood_scale=args.likelihood_scale, percentiles=percentiles,
                     results_dict=hmc_result_dict, MAP_params=MAP_params, random_mask=args.random_mask)
