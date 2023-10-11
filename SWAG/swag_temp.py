import copy

import torch
import numpy as np
import pickle
import torch.nn as nn
from torch.optim import SGD
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import SWAG.utils_temp as utils_temp
import misc.likelihood_losses as ll


def run_swag_partial(model:nn.Module,
                     train_loader: DataLoader,
                     lr=1e-2,
                     momentum=0.9,
                     weight_decay=3e-4,
                     K=100,
                     n_epochs=10,
                     snapshots_per_epoch=4,
                     criterion = nn.CrossEntropyLoss,
                     mask = None):
    """

    :param bnn_params: (key, value) = (str, [str,..]) where the key is the module, and the list contains the names of
    the submodules, if the list is empty, the entire module is set to not require a gradient, else each subcomponent
    :param model: fully trained model (nn.module)
    :param train_loader: dataloader for training data, expected to give (x, label) from getitem
    :param lr: Learning rate, has to be put high
    :param momentum:
    :param weight_decay: Important to maintain Gaussian distribution, dont set to zero
    :param K: Number of columns in the deviation matrix
    :param n_epochs: number of epochs, makes pretty good sense what this is
    :param snapshots_per_epoch: how many in an epoch we want to save the  weights
    :return:
    """

    batch_snapshot_freq = int(len(train_loader) / snapshots_per_epoch)

    _model = deepcopy(model)
    _model.eval()
    device = next(_model.parameters()).device

    mean = torch.zeros_like(utils_temp._parameter_vector(_model))
    sq_mean = torch.zeros_like(utils_temp._parameter_vector(_model))
    if mask is None:
        mask = torch.ones_like(mean)

    mean = mean[mask]
    sq_mean = sq_mean[mask]

    deviations = []

    n_snapshots = 0

    optimizer = SGD(
        [p for p in _model.parameters() if p.requires_grad], lr = lr, momentum=momentum, weight_decay = weight_decay
    )


    average_losses = []
    pbar = tqdm(range(n_epochs))
    for _ in pbar:
        def snapshot(n_snapshots_lambda, model_lambda, mean_lambda, sq_mean_lambda, deviations_lambda, K):

            old_fac, new_fac = n_snapshots_lambda / (n_snapshots_lambda + 1), 1 / (n_snapshots_lambda + 1)

            mean_lambda = mean_lambda * old_fac + utils_temp._parameter_vector(model_lambda)[mask] * new_fac
            sq_mean_lambda = sq_mean_lambda * old_fac + utils_temp._parameter_vector(model_lambda)[mask] ** 2 * new_fac
            deviation = utils_temp._parameter_vector(model_lambda)[mask] - mean_lambda

            if len(deviations_lambda) == K:
                deviations_lambda.pop(0)
            deviations_lambda.append(deviation)

            return n_snapshots_lambda + 1, mean_lambda, sq_mean_lambda, deviations_lambda

        epoch_losses = []
        for batch_i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if isinstance(criterion, ll.BaseMAPLossSwag):
                loss = criterion(_model(inputs), targets, _model)
            else:
                loss = criterion(_model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.cpu().clone().detach().numpy())

            if batch_i % batch_snapshot_freq == (batch_snapshot_freq - 1):
                n_snapshots, mean, sq_mean, deviations = snapshot(n_snapshots, _model, mean, sq_mean, deviations, K)
                pbar.set_postfix({'Running epoch_loss': np.mean(epoch_losses), 'snapshots': n_snapshots})

        average_losses.append(np.mean(epoch_losses))

    D = torch.zeros((utils_temp._parameter_vector(model)[mask].numel(), K))
    for i in range(K):
        D[:, i] = deviations[i]

    return {
        "theta_swa": mean.to(device),
        "sigma_diag": (sq_mean - mean ** 2).to(device).clamp(1e-14),
        "D": D.to(device),
        "K": K
    }


class TestModelSmall(nn.Module):
    def __init__(self):
        super(TestModelSmall, self).__init__()

        self.layer_one = nn.Linear(10, 10)
        self.activation = nn.ReLU()
        self.layer_two = nn.Linear(10, 10)

    def forward(self, x):

        out = self.layer_one(x)
        out = self.activation(out)
        out = self.layer_two(out)
        return out



class TestModelMulti(nn.Module):
    def __init__(self):
        super(TestModelMulti, self).__init__()

        self.module_one = TestModelSmall()
        self.module_two = TestModelSmall()
        self.activation = nn.ReLU()
        self.last_layer = nn.Linear(10, 2)
    def forward(self, x):
        out = self.module_one(x)
        out = self.activation(out)
        out = self.module_two(out)
        out = self.activation(out)
        out = self.last_layer(out)

        return out


class TestData(Dataset):

    def __init__(self):

        self.data = np.random.normal(0,1, (1000, 10))
        labels_ = np.random.binomial(n = 2, p = 0.5, size = (1000, ))
        self.labels = [[1, 0] if lab == 0 else [0, 1] for lab in labels_ ]

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):

        data = torch.from_numpy(self.data[item])
        label = torch.Tensor(self.labels[item])
        data = data.float()
        label = label.float()
        return data, label


def test_swag():

    model = TestModelMulti()
    partial_dict = {'module_one': ['layer_one']}

    utils_temp.set_gradient_parameters(model, partial_dict)
    criterion = nn.CrossEntropyLoss()
    dataset = TestData()
    dataloader = DataLoader(dataset, batch_size=10)

    swag_res = run_swag_partial(model, dataloader,n_epochs=40, K = 10)
    breakpoint()


if __name__ == '__main__':

    test_swag()
    breakpoint()