import copy

import torch
import numpy as np
import pickle
import torch.nn as nn
from torch.optim import SGD
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import utils_temp
from tqdm import tqdm
def run_swag_partial(model:nn.Module,
                     train_loader: DataLoader,
                     lr=1e-2,
                     momentum=0.9,
                     weight_decay=3e-4,
                     K=20,
                     n_epochs=10,
                     snapshots_per_epoch=4,
                     criterion = nn.CrossEntropyLoss,
                     bnn_params=None):
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
    if bnn_params is None:
        UserWarning("bnn_params set to none, running full swag")
        model.requires_grad_(True)
    else:
        utils_temp.set_gradient_parameters(bnn_params)

    batch_snapshot_freq = int(len(train_loader) / snapshots_per_epoch)

    _model = deepcopy(model)
    _model.eval()
    device = next(_model.parameters()).device

    mean = torch.zeros_like(utils_temp._parameter_vector(_model))
    sq_mean = torch.zeros_like(utils_temp._parameter_vector(_model))
    deviations = []

    n_snapshots = 0

    optimizer = SGD(
        [p for p in model.parameters() if p.requires_grad], lr = lr, momentum=momentum, weight_decay = weight_decay
    )

    criterion = criterion()
    average_losses = []
    pbar = tqdm(range(n_epochs))
    for _ in pbar:
        def snapshot(n_snapshots_lambda, model_lambda, mean_lambda, sq_mean_lambda, deviations_lambda, K):

            old_fac, new_fac = n_snapshots_lambda / (n_snapshots_lambda + 1), 1 / (n_snapshots_lambda + 1)
            mean_lambda = mean_lambda * old_fac + utils_temp._parameter_vector(model_lambda) * new_fac
            sq_mean_lambda = sq_mean_lambda * old_fac + utils_temp._parameter_vector(model_lambda) ** 2 * new_fac
            deviation = utils_temp._parameter_vector(model_lambda) - mean_lambda

            if len(deviations_lambda) == K:
                deviations_lambda.pop(0)
            deviations_lambda.append(deviation)

            return n_snapshots_lambda + 1, mean_lambda, sq_mean_lambda, deviations_lambda

        epoch_losses = []
        for batch_i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(_model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.cpu().clone().detach().numpy())

            if batch_i % batch_snapshot_freq == (batch_snapshot_freq - 1):
                n_snapshots, mean, sq_mean, deviations = snapshot(n_snapshots, _model, mean, sq_mean, deviations, K)
                pbar.set_postfix({'Running epoch_loss': np.mean(epoch_losses), 'snapshots': n_snapshots})

        average_losses.append(np.mean(epoch_losses))
