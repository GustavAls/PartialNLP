import sys, os, time, requests
import argparse
import numpy as np
import pandas as pd
import torch.distributions
from MAP_baseline import trainer
from MAP_baseline.MapNN import MapNN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, Trace_ELBO, autoguide, SVI
import jax
import jax.numpy as jnp
import jax.nn
from VI.partial_bnn_functional import *
from jax import random


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
class UCIDataset():
    def __init__(
        self,
        data_dir: str,
        test_split_type: str ="random",
        gap_column: int = 0,
        test_size: float = 0.2,
        val_fraction_of_train: float = 0.1,
        seed:int = 42,
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
            self.data_dir, "uci_datasets", self.dataset_name, self.filename
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


class UCIDataloader(Dataset):
    def __init__(self, X, y):
        super(UCIDataloader, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return torch.tensor(self.X[item], dtype=torch.float32), torch.tensor(self.y[item], dtype=torch.float32)


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
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    with numpyro.handlers.scale(scale=scale):
        y_obs = numpyro.sample("y_obs", dist.Normal(mean, sigma_obs), obs=y)


def evaluate_MAP_old(model, svi_results, X, y, rng_key, y_scale=1.0, y_loc=0.0):
    predictive = Predictive(
        model=model,
        guide=autoguide.AutoDelta(model),
        params=svi_results.params,
        num_samples=1,
    )(rng_key, X=X)

    sigma_obs = 1.0 / jnp.sqrt(svi_results.params["prec_obs_auto_loc"])
    rmse = (((predictive["mean"][0, :] - y) ** 2).mean() ** 0.5) * y_scale
    log_likelihood = (
        dist.Normal((y_scale * predictive["mean"][0, :]) + y_loc, sigma_obs * y_scale)
        .log_prob(y_loc + (y * y_scale))
        .mean()
    )

    return float(log_likelihood), float(rmse)


# def evaluate_MAP(model, X, y, y_scale=1.0, y_loc=0.0):
#     y_scale = torch.tensor(y_scale)
#     y_loc = torch.tensor(y_loc)
#     model = model.to("cpu")
#     y_pred = model(torch.tensor(X, dtype=torch.float32))
#     rmse = (((y_pred.detach().numpy() - y) ** 2).mean() ** 0.5) * y_scale
#     y = torch.tensor(y)
#
#     # log_likelihood = (torch.distributions.Normal((y_scale * y_pred) + y_loc, sigma_obs * y_scale).log_prob(y_loc + (y * y_scale)).mean())
#     loss = nn.GaussianNLLLoss()
#     loss(torch.from_numpy(np.array(predictive["mean"][0, :])), torch.from_numpy(y),
#          torch.ones_like(torch.from_numpy(y)) * np.var(y - np.array(predictive["mean"][0, :])).item())
#     return float(log_likelihood), float(rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--dataset", type=str, default="boston")
    parser.add_argument("--gap", action="store_true", default=True)
    parser.add_argument("--prior_variance", type=float, default=0.1) #0.1 is good for yacht, but not for other datasets

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

    dataset = dataset_class(data_dir=os.getcwd(),
                            test_split_type="gap" if args.gap else "random",
                            test_size=0.1,
                            gap_column=1,
                            val_fraction_of_train=0.1,
                            seed=args.seed)

    n_train, p = dataset.X_train.shape
    n_val = dataset.X_val.shape[0]
    out_dim = dataset.y_train.shape[1]

    train_args = {'epochs': 2000,
                  'device': 'cpu',
                  'save_path': os.path.join(os.getcwd(), "UCI_models_gap\\" + args.dataset)}
    percentages = [1, 2, 5, 8, 14, 23, 37, 61, 100]
    train_dataloader = DataLoader(UCIDataloader(dataset.X_train, dataset.y_train), batch_size=n_train)
    val_dataloader = DataLoader(UCIDataloader(dataset.X_val, dataset.y_val), batch_size=n_val)

    ### Train MAP Solution
    # optimizer = numpyro.optim.Adam(0.01)
    # rng_key = random.PRNGKey(1)
    #
    # model = lambda X, y=None: one_d_bnn(X, y, prior_variance=args.prior_variance)
    #
    # svi = SVI(model, autoguide.AutoDelta(one_d_bnn), optimizer, Trace_ELBO())
    # start_time = time.time()
    # svi_results = svi.run(rng_key, 1000, X=dataset.X_train, y=dataset.y_train)
    #
    # train_ll, train_rmse = evaluate_MAP_old(
    #     model,
    #     svi_results,
    #     dataset.X_train,
    #     dataset.y_train,
    #     rng_key,
    #     y_scale=dataset.scl_Y.scale_,
    #     y_loc=dataset.scl_Y.mean_,
    # )

    train_model_with_varying_stochasticity_scheme_two(MapNN(p, 50, 2, out_dim, "leaky_relu"),
                                                      train_dataloader,
                                                      val_dataloader,
                                                      percentages,
                                                      train_args)


    # train_model_with_varying_stochasticity(untrained_model=model,
    #                                        dataloader=train_dataloader,
    #                                        dataloader_val=val_dataloader,
    #                                        percentages=percentages,
    #                                        train_args=train_args)

    end_time = time.time()

    # train_ll, train_rmse = evaluate_MAP(
    #     model,
    #     dataset.X_test,
    #     dataset.y_test,
    #     y_scale=dataset.scl_Y.scale_,
    #     y_loc=dataset.scl_Y.mean_
    # )
    #
    # val_ll, val_rmse = evaluate_MAP(
    #     model,
    #     dataset.X_val,
    #     dataset.y_val,
    #     y_scale=dataset.scl_Y.scale_,
    #     y_loc=dataset.scl_Y.mean_
    # )
    # test_ll, test_rmse = evaluate_MAP(
    #     model,
    #     dataset.X_test,
    #     dataset.y_test,
    #     y_scale=dataset.scl_Y.scale_,
    #     y_loc=dataset.scl_Y.mean_
    # )
    #
    # map_results = {
    #     "prior_variance": args.prior_variance,
    #     "test_rmse": test_rmse,
    #     "test_ll": test_ll,
    #     "val_rmse": val_rmse,
    #     "val_ll": val_ll,
    #     "train_rmse": train_rmse,
    #     "train_ll": train_ll,
    #     "runtime": end_time - start_time,
    #     "num_params_sampled": 0,
    #     "dataset": args.dataset,
    #     "seed": args.seed,
    #     "gap_split?": args.gap,
    #     "name": "MAP",
    # }
    #
    # print(map_results)
    #
    # all_results = {
    #     "all_results_not_scaled": [map_results, map_results],
    #     "all_results_scaled": [map_results, map_results],
    # }
    #
    # file_name = os.path.join(args.output_path, "uci_results/map_" + args.dataset)
    #
    # pickle.dump(all_results, open(f"{file_name}.pkl", "wb"))

