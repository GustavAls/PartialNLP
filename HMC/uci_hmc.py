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
from Laplace.uci_laplace import calculate_std
from jax import random
numpyro.set_platform("cpu")
numpyro.set_host_device_count(8)
import argparse
from torch.distributions import Normal


def tensor_to_jax_array(tensor):
    return jnp.array(tensor.cpu().detach().numpy())


def convert_torch_to_pyro_params(torch_params, MAP_params, precision):
    # Torch model does not have a precision parameter
    assert len(MAP_params.keys()) - 1 == len(torch_params.keys())
    for svi_key in MAP_params.keys():
        # Hardcoded for now
        if "W1" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['linear1.weight'].detach()).T
        elif "W2" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['linear2.weight'].detach()).T
        elif "b1" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['linear1.bias'].detach()).T
        elif "b2" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['linear2.bias'].detach()).T
        elif "W_output" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['out.weight'].detach()).T
        elif "b_output" in svi_key:
            MAP_params[svi_key] = tensor_to_jax_array(torch_params['out.bias'].detach()).T
        # elif "prec_obs_auto_loc" in svi_key:
        #     MAP_params[svi_key] = jnp.array(precision)

    return MAP_params


def _gap_train_test_split(X, y, gap_column, test_size):
    n_data = X.shape[0]
    sorted_idxs = np.argsort(X[:, gap_column])
    train_idxs = np.concatenate(
        (
            sorted_idxs[: int(n_data * 0.5 * (1 - test_size))],
            sorted_idxs[-int(n_data * 0.5 * (1 - test_size)) :],
        )
    )
    test_idxs = np.array(list(set(sorted_idxs.tolist()) - set(train_idxs.tolist())))
    X_train = X[train_idxs, :]
    X_test = X[test_idxs, :]
    y_train = y[train_idxs, :]
    y_test = y[test_idxs, :]
    return X_train, X_test, y_train, y_test


### dataset stuff


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
            f"Train set: {n_train} examples, {100*(n_train/n_total):.2f}% of all examples."
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


def evaluate_samples_properly(model, rng_key, X, y, samples, y_scale=1.0, y_loc=0.0):
    sigma_obs = (1.0 / jnp.sqrt(samples["prec_obs"])).mean()

    predictive = Predictive(model, samples)(rng_key, X=X)

    predictive_mean = np.array(predictive['mean']).squeeze(-1)
    y = y.squeeze(-1)

    likelihood = calculate_ll_mc(y, predictive_mean)
    return likelihood


def evaluate_MAP(model, MAP_params, X, y, rng_key, y_scale=1.0, y_loc=0.0):
    predictive = Predictive(
        model=model,
        guide=autoguide.AutoNormal(model),
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
    MAP_params, sample_mask_tuple, prior_variance, scale=1.0
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
            "b1_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width)))
        )
        W_2_noise = numpyro.sample(
            "W2_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, width))),
        )
        b_2_noise = numpyro.sample(
            "b2_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width)))
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
        prec_obs = numpyro.sample(
            "prec_obs", dist.Gamma(3.0, 1.0)
        )  # MAP outperforms full BNN, even if we freeze the prior precision. That's interesting here, I think.
        sigma_obs = 1.0 / jnp.sqrt(prec_obs)

        with numpyro.handlers.scale(scale=scale):
            y_obs = numpyro.sample("y_obs", dist.Normal(mean, sigma_obs), obs=y)

    return mixed_bnn


def calculate_ll_mc(labels, mc_matrix, sigma, y_scale, y_loc):
    results = []

    for i in range(mc_matrix.shape[1]):
        res_temp = []
        for j in range(mc_matrix.shape[0]):
            dist = Normal(mc_matrix[j,i].item() * y_scale + y_loc, sigma * y_scale)
            res_temp.append(dist.log_prob(torch.tensor([labels[i].item()]) * y_scale + y_loc).item())
        results.append(np.mean(res_temp))
    return np.mean(results)


def calculate_ll_ours(model, params, dataset, bnn, num_mc_samples = 200, delta = False, mle_model=None):
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

    predictive_test = Predictive(
        model=model,
        guide=guide(),
        params=params,
        num_samples=num_mc_samples,
    )(rng_key, X=dataset.X_test)

    predictive_val = Predictive(
        model=model,
        guide=guide(),
        params=params,
        num_samples=num_mc_samples,
    )(rng_key, X=dataset.X_val)
    y_scale = dataset.scl_Y.scale_
    y_loc = dataset.scl_Y.mean_
    ytrain, yval, ytest = dataset.y_train.squeeze(), dataset.y_val.squeeze(), dataset.y_test.squeeze()

    # TODO: For evaluating with MLE and other nll method, does not give reliable results atm
    if num_mc_samples == 1 and mle_model is not None:
        ptrain = mle_model(torch.tensor(dataset.X_train, dtype=torch.float32)).detach().numpy()
        ptest = mle_model(torch.tensor(dataset.X_test, dtype=torch.float32)).detach().numpy()
        pval = mle_model(torch.tensor(dataset.X_val, dtype=torch.float32)).detach().numpy()
        train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train // 8)
        sigma = calculate_std(mle_model, train_dataloader, alpha=3, beta=1, beta_prior=False)

    if num_mc_samples == 1:
        ptrain = np.asarray(predictive_train['mean'].squeeze(0))
        ptest =  np.asarray(predictive_test['mean'].squeeze(0))
        pval = np.asarray(predictive_val['mean'].squeeze(0))
        sigma = np.std(ptrain - ytrain)
    else:
        ptrain = np.asarray(predictive_train['mean'].squeeze(-1))
        ptest =  np.asarray(predictive_test['mean'].squeeze(-1))
        pval = np.asarray(predictive_val['mean'].squeeze(-1))
        sigma = np.std(ptrain - ytrain)

    if num_mc_samples == 1 and mle_model is not None:
        test_nll = calculate_nll(torch.tensor(ptest), torch.tensor(ytest),
                                 torch.tile(torch.tensor(sigma), (len(ytest),)), y_scale.item(), y_loc.item())
        val_nll = calculate_nll(torch.tensor(pval), torch.tensor(yval),
                                torch.tile(torch.tensor(sigma), (len(yval),)), y_scale.item(), y_loc.item())

    else:
        test_nll = -calculate_ll_mc(ytest, ptest, sigma, y_scale.item(), y_loc.item())
        val_nll = -calculate_ll_mc(yval, pval, sigma, y_scale.item(), y_loc.item())


    return -test_nll, -val_nll


def calculate_nll(preds, labels, sigma, y_scale, y_loc):
    """Calculate the negative log likelihood of the predictions.
        Args:
            preds: (np.array) predictions of the model
            label: (np.array) true labels
            sigma: (float) standard deviation of the predictions
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



def create_sample_mask_largest_abs_values(percentile, MAP_params):
    keys = [
        "W1_auto_loc",
        "W2_auto_loc",
        "W_output_auto_loc",
        "b1_auto_loc",
        "b2_auto_loc",
        "b_output_auto_loc",
    ]

    all_values = np.concatenate([MAP_params[key].ravel() for key in keys])
    param_abs_values = np.abs(all_values)
    val = np.percentile(param_abs_values, 100 - percentile)

    W1_sample_mask = np.abs(MAP_params["W1_auto_loc"]) >= val
    W2_sample_mask = np.abs(MAP_params["W2_auto_loc"]) >= val
    W_output_sample_mask = np.abs(MAP_params["W_output_auto_loc"]) >= val
    b1_sample_mask = np.abs(MAP_params["b1_auto_loc"]) >= val
    b2_sample_mask = np.abs(MAP_params["b2_auto_loc"]) >= val
    b_output_sample_mask = np.abs(MAP_params["b_output_auto_loc"]) >= val

    sample_mask_tuple = (
        W1_sample_mask,
        W2_sample_mask,
        W_output_sample_mask,
        b1_sample_mask,
        b2_sample_mask,
        b_output_sample_mask,
    )

    return sample_mask_tuple


def run_for_percentile(
    dataset,
    percentile,
    MAP_params,
    prior_variance=0.8,
    prior_variance_scaled=True,
    scale=1.0
):
    sample_mask_tuple = create_sample_mask_largest_abs_values(percentile, MAP_params)
    prior_variance_used = (
        prior_variance
        if not prior_variance_scaled
        else prior_variance * (100 / percentile)
    )

    mixed_bnn = generate_mixed_bnn_by_param(
        MAP_params,
        create_sample_mask_largest_abs_values(percentile, MAP_params),
        prior_variance_used,
        scale=scale,
    )

    nuts_kernel = NUTS(mixed_bnn, max_tree_depth=15)
    mcmc = MCMC(nuts_kernel, num_warmup=325, num_samples=75, num_chains=8)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, dataset.X_train, dataset.y_train)
    test_ll_ours = evaluate_samples_properly(mixed_bnn, rng_key, dataset.X_test, dataset.y_test,
                                             mcmc.get_samples(), y_scale=dataset.scl_Y.scale_, y_loc=dataset.scl_Y.mean_)

    test_ll_theirs, _ = evaluate_samples(mixed_bnn, rng_key, dataset.X_test, dataset.y_test,
                                      mcmc.get_samples(), y_scale=dataset.scl_Y.scale_, y_loc=dataset.scl_Y.mean_)

    results = {
        "prior_variance": prior_variance_used,
        "test_ll_ours": test_ll_ours,
        "test_ll_theirs": test_ll_theirs,
        "num_params_sampled": np.array([t.sum() for t in sample_mask_tuple]).sum(),
        "dataset": args.dataset,
        "seed": rand_seed,
        "prior_variance_scaled": True,
        "scale": scale,
    }

    return results


def make_vi_run(run, dataset, prior_variance, scale, results_dict, save_path, model, MAP_params, mle_model=None):

    percentiles = [1, 2, 5, 8, 14, 23, 37, 61, 100]
    test_ll_theirs, _ = evaluate_MAP(model, MAP_params, dataset.X_test, dataset.y_test, rng_key, y_scale=dataset.scl_Y.scale_, y_loc=dataset.scl_Y.mean_)
    test_ll_ours, val_ll_ours = calculate_ll_ours(model, MAP_params, dataset, one_d_bnn, num_mc_samples=1, mle_model=mle_model)
    results_dict['test_ll_ours'].append(test_ll_ours)
    results_dict['val_ll_ours'].append(val_ll_ours)
    results_dict['test_ll_theirs'].append(test_ll_theirs)
    results_dict['percentiles'] = [0] + percentiles

    for percentile in percentiles:
        sample_mask_tuple = create_sample_mask_largest_abs_values(percentile, MAP_params)
        optimizer = numpyro.optim.Adam(0.01)

        mixed_bnn = generate_mixed_bnn_by_param(
            MAP_params,
            create_sample_mask_largest_abs_values(percentile, MAP_params),
            prior_variance,
            scale=scale,
        )
        model = lambda X, y=None: generate_mixed_bnn_by_param(
            MAP_params,
            sample_mask_tuple,
            prior_variance,
            scale=scale,
        )(X, y)

        svi=SVI(model, autoguide.AutoNormal(mixed_bnn), optimizer, Trace_ELBO())
        svi_results = svi.run(rng_key, 20000, X=dataset.X_train, y=dataset.y_train)

        test_ll_ours, val_ll_ours = calculate_ll_ours(model, svi_results.params, dataset, mixed_bnn, delta=False)
        test_ll_theirs, _ = evaluate_samples(mixed_bnn, rng_key, dataset.X_test, dataset.y_test,
                                          mcmc.get_samples(), y_scale=dataset.scl_Y.scale_, y_loc=dataset.scl_Y.mean_)

        results_dict['test_ll_ours'].append(test_ll_ours)
        results_dict['val_ll_ours'].append(val_ll_ours)
        results_dict['test_ll_theirs'].append(test_ll_theirs)

    save_name = f'results_vi_run_{run}.pkl'
    with open(os.path.join(save_path, save_name), 'wb') as handle:
        pickle.dump(results_dict, handle,protocol=pickle.HIGHEST_PROTOCOL)


def train_MAP_solution(mle_model, dataset, num_epochs):
    loss_arguments = {'loss': GLLGP_loss_swag , 'prior_sigma': 1}
    if issubclass(loss := loss_arguments.get('loss', MSELoss), BaseMAPLossSwag):
        loss_arguments['model'] = mle_model
        loss_fn = loss(**loss_arguments)

    train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train // 8)
    val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)

    mle_model = train(network=mle_model, dataloader_train=train_dataloader, dataloader_val=val_dataloader,
                      model_old=None, vi=False, device='cpu', epochs=num_epochs, return_best_model=True, criterion=loss_fn)
    return mle_model


def make_hmc_run(run, dataset, scale_prior, prior_variance, save_path, likelihood_scale, percentiles, results_dict):
    MAP_params = results_dict['map_results']['map_params']
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
            )
            print(results_dict[f"{percentile}"])
            pickle.dump(results_dict, open(os.path.join(save_path, f"results_hmc_run_{run}.pkl"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dataset", type=str, default="boston")
    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--data_path", type=str, default=os.getcwd())
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--run", type=int, default=15)
    parser.add_argument("--num_epochs", type=int, default=20000)
    parser.add_argument("--scale_prior",  type=ast.literal_eval, default=True)
    parser.add_argument("--prior_variance", type=float, default=1.0) #0.1 is good for yacht, but not for other datasets
    parser.add_argument("--likelihood_scale", type=float, default=1.0) #6.0 is good for yacht, but not for other datasets
    parser.add_argument('--vi', type=ast.literal_eval, default=True)
    args = parser.parse_args()

    if args.dataset == "yacht":
        dataset_class = UCIYachtDataset
    elif args.dataset == "energy":
        dataset_class = UCIEnergyDataset
    elif args.dataset == "boston":
        dataset_class = UCIBostonDataset

    percentiles = [1, 2, 5, 8, 14, 23, 37, 61, 100]

    rand_seed = np.random.randint(0, 10000)
    dataset = dataset_class(
        args.data_path,
        seed=rand_seed,
        test_split_type="random",
        test_size=0.1,
        val_fraction_of_train=0.1,
    )
    ### Train MAP Solution
    if os.path.exists(os.path.join(args.output_path, f"results_hmc_run_{args.run}.pkl")):
        hmc_result_dict = pickle.load(open(os.path.join(args.output_path, f"results_hmc_run_{args.run}.pkl"), "rb"))
    else:
        rng_key = random.PRNGKey(1)
        optimizer = numpyro.optim.Adam(0.01)
        model = lambda X, y=None: one_d_bnn(X, y, prior_variance=args.prior_variance)

        svi = SVI(model, autoguide.AutoDelta(one_d_bnn), optimizer, Trace_ELBO())
        svi_results = svi.run(rng_key, args.num_epochs, X=dataset.X_train, y=dataset.y_train)
        MAP_params = svi_results.params
        # Overwrite MAP params with the ones from the saved model

        if args.map_path is None:
            mle_model = None
        else:
            # TODO: map model as start of HMC/VI
            # When completed model is ready
            n_train, p = dataset.X_train.shape
            n_val = dataset.X_val.shape[0]
            out_dim = dataset.y_train.shape[1]
            mle_model = MapNN(input_size=p, width=50, output_size=out_dim, non_linearity="leaky_relu")
            mle_model.load_state_dict(torch.load(args.map_path))
            # Testing with MAP solution
            # mle_model = train_MAP_solution(mle_model, dataset, args.num_epochs)
            mle_state_dict = mle_model.state_dict()
            # train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train // 8)
            # sigma = calculate_std(mle_model, train_dataloader, alpha=3, beta=1, beta_prior=False)
            # precision = 1 / (sigma ** 2)
            precision = None
            MAP_params = convert_torch_to_pyro_params(mle_state_dict, MAP_params, precision)

        vi_results_dict = {'percentiles': None, 'test_ll_ours': [], 'val_ll_ours': [], 'test_ll_theirs': []}

        test_ll_theirs, _ = evaluate_MAP(model, MAP_params, dataset.X_test, dataset.y_test,
                                         rng_key, y_scale=dataset.scl_Y.scale_, y_loc=dataset.scl_Y.mean_)

        test_ll_ours, val_ll_ours = calculate_ll_ours(model, MAP_params, dataset, one_d_bnn, num_mc_samples=1, mle_model=mle_model)

        vi_results_dict['map_params'] = MAP_params
        vi_results_dict['test_ll_ours'].append(test_ll_ours)
        vi_results_dict['val_ll_ours'].append(val_ll_ours)
        vi_results_dict['test_ll_theirs'].append(test_ll_theirs)
        vi_results_dict['percentiles'] = [0] + percentiles

        hmc_result_dict = {'map_results': {'map_params': MAP_params,
                                           'test_ll_ours': test_ll_ours,
                                           'test_ll_theirs': test_ll_theirs,
                                           'val_ll_ours': val_ll_ours}}
        pickle.dump(vi_results_dict, open(os.path.join(args.output_path, f"results_vi_run_{args.run}.pkl"), "wb"))
        pickle.dump(hmc_result_dict, open(os.path.join(args.output_path, f"results_hmc_run_{args.run}.pkl"), "wb"))

        if args.vi:
            make_vi_run(args.run, dataset, args.prior_variance, args.likelihood_scale, vi_results_dict,
                        save_path=args.output_path, model=model, MAP_params=MAP_params, mle_model=mle_model)

    make_hmc_run(args.run, dataset, args.scale_prior, args.prior_variance,
                 args.output_path, args.likelihood_scale, percentiles,  hmc_result_dict)

