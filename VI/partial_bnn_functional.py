import copy
from typing import List, Any

import torch
import numpy as np
import pickle
import torch.nn as nn
from torch.optim import SGD
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torchvision
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import SGD, Adam
import torch.nn as nn
import os
from copy import deepcopy
from tqdm import trange
import os


def bnn(model, mask=None):
    if mask is None:
        mask = {}
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }

    dnn_to_bnn(model, const_bnn_prior_parameters, mask)
    return None


def vector_mask_to_parameter_mask(vec, parameters) -> list[Any]:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    param_masks = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param_masks.append(vec[pointer:pointer + num_param].view_as(param).data)
        # Increment the pointer
        pointer += num_param

    return param_masks


def create_mask(model, percentile):
    parameter_vector = nn.utils.parameters_to_vector(model.parameters())
    argsorted = torch.argsort(torch.abs(parameter_vector), descending=True)

    bnn_length = int(len(argsorted) * percentile / 100)
    mask = torch.zeros_like(parameter_vector)
    mask[argsorted[:bnn_length]] = 1

    param_mask = vector_mask_to_parameter_mask(mask, model.parameters())
    param_mask = order_with_bias(param_mask, model)
    return param_mask


def order_with_bias(param_mask, model):
    name_to_mask = {}
    counter = 0
    for name, module in list(model._modules.items()):
        if "Linear" in model._modules[name].__class__.__name__:
            if module.bias is not None:
                name_to_mask[name] = [param_mask[counter], param_mask[counter + 1]]
                counter += 2
            else:
                name_to_mask[name] = [param_mask[counter]]
                counter += 1
    return name_to_mask


def set_model_weights(model, model_, name_to_mask):
    org_model_items = list(model_._modules.items())
    new_model_items = list(model._modules.items())

    for (name_, module_), (name, module) in zip(org_model_items, new_model_items):
        if hasattr(module_, 'bias'):
            if module_.bias is not None:
                mask_linear = ~(name_to_mask[name_][0] == 1)
                mask_bias = ~(name_to_mask[name_][1] == 1)
                parameters = [p for p in module_.parameters()]
                module.mu_weight.data[mask_linear] = parameters[0][mask_linear].data.clone()
                module.mu_bias.data[mask_bias] = parameters[1][mask_bias].data.clone()

            else:
                mask_linear = ~(name_to_mask[name_][0] == 1)
                parameters = [p for p in module_.parameters()]
                module.mu_weight.data[mask_linear] = parameters[0][mask_linear].data.clone()

    return None


def train(network: nn.Module,
          dataloader_train,
          dataloader_val,
          model_old=None,
          mask=None,
          vi=True,
          device='cpu',
          epochs=50,
          save_path=None,
          num_mc_samples = 25):
    """

    :param network: (nn.Module) feed forward classification model
    :param dataloader_train: dataloader for the training cases, should output (inputs, labels)
    :param dataloader_val: dataloader for the validation cases, should output (inputs, labels)
    :return: The trained model (the best version of the trained model, from eval on validation set)
    """

    network.to(device)
    optimizer = SGD(network.parameters(), lr=0.1)

    loss_fn = nn.MSELoss()
    best_loss = np.infty
    best_model = None
    if vi:
        current_loss = evaluate_monte_carlo(network, dataloader_val, loss_fn, num_mc_samples, device)
        print(f'"loss without training was {current_loss}')

    for epoch in trange(epochs, desc="Training MAP network"):
        network.train()
        for idx, (batch, target) in enumerate(dataloader_train):
            optimizer.zero_grad()
            batch = batch.to(device)
            target = target.to(device)
            output = network(batch)
            loss = loss_fn(output, target)
            if vi:
                kl = get_kl_loss(network)
                loss += kl / batch.shape[0]

            loss.backward()
            optimizer.step()
            if model_old is not None:
                set_model_weights(network, model_old, mask)

        if epoch % 2 == 0:
            if vi:
                current_loss = evaluate_monte_carlo(network, dataloader_val,loss_fn, num_mc_samples, device)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model = deepcopy(network)
                continue

            with torch.no_grad():
                network.eval()
                current_loss = 0
                for idx, (batch, target) in enumerate(dataloader_val):
                    batch = batch.to(device)
                    output = network(batch)
                    current_loss += loss_fn(output, target.to(device))
                current_loss /= len(dataloader_val)

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model = deepcopy(network)

    if best_model is None:
        UserWarning("The model failed to improve, something went wrong")
    else:

        torch.save(best_model.state_dict(), save_path)
        print(f"Model was saved to location {save_path}, terminated with MSELoss {best_loss}")

    return best_model

def get_sigma(model, dataloader, vi = True, num_mc_samples = 25, device = 'cpu'):

    predictions = []
    targets = []
    rmse = lambda pred, target: ((pred - target)**2).mean()**0.5
    with torch.no_grad():
        for idx, (batch, target) in enumerate(dataloader):
            mc_output = []
            batch = batch.to(device)
            target = target.to(device)
            if vi:
                for _ in range(num_mc_samples):
                    mc_output.append(model(batch))
                predictions.append(torch.stack(mc_output).mean(0))
            else:
                predictions.append(model(batch))
            targets.append(target)

        if len(targets) > 1:
            targets = torch.stack(targets)
            predictions = torch.stack(predictions)
        else:
            targets = targets[0]
            predictions = predictions[0]
    return rmse(predictions, targets)

def evaluate_monte_carlo(model, dataloader, loss_fn, num_mc_samples = 25, device = 'cpu'):

    with torch.no_grad():
        loss = 0
        mc_output = []
        for idx, (batch, target) in enumerate(dataloader):
            for mc_run in range(num_mc_samples):
                batch = batch.to(device)
                output = model(batch)
                mc_output.append(output)

            loss += loss_fn(torch.stack(mc_output).mean(0), target.to(device))
        loss /= len(dataloader)

        return loss

def train_model_with_varying_stochasticity(untrained_model, dataloader, dataloader_val, percentages, train_args):
    model_ = train(
        untrained_model,
        dataloader,
        dataloader_val,
        model_old=None,
        vi=False,
        device=train_args['device'],
        epochs=train_args['epochs'],
        save_path=os.path.join(train_args['save_path'], "map_model.pt")
    )

    model = copy.deepcopy(model_)
    model = model.to(train_args['device'])
    for percentage in percentages:
        mask = create_mask(model_, percentage)
        bnn(model, mask)
        set_model_weights(model, model_, mask)

        save_path = os.path.join(train_args['save_path'], f"model_with_{percentage}_pct_stoch.pt")
        model = train(
            network=model,
            dataloader_train=dataloader,
            dataloader_val=dataloader_val,
            model_old=model_,
            mask=mask,
            device=train_args['device'],
            epochs=train_args['epochs'],
            save_path=save_path
        )

def get_preds(model, dataloader, num_mc_samples = 100, device = 'cpu'):

    predictions, labels = [], []
    with torch.no_grad():
        mc_output = []
        for idx, (batch, label) in enumerate(dataloader):
            for mc_run in range(num_mc_samples):
                batch = batch.to(device)
                output = model(batch)
                mc_output.append(output)

            predictions.append(torch.stack(mc_output).mean(0))
            labels.append(label)

    return predictions, labels

def train_model_with_varying_stochasticity_scheme_two(
        uninitialised_model,
        dataloader,
        dataloader_val,
        percentages,
        train_args,
        run_number = 0,
        dataloader_test = None
):

    untrained_model = uninitialised_model

    model_ = train(
        copy.deepcopy(untrained_model),
        dataloader,
        dataloader_val,
        model_old=None,
        vi=False,
        device=train_args['device'],
        epochs=train_args['epochs'],
        save_path=os.path.join(train_args['save_path'], f"map_model_run_{run_number}.pt")
    )


    for percentage in percentages:
        model = copy.deepcopy(untrained_model)
        model = model.to(train_args['device'])
        mask = create_mask(model_, percentage)
        bnn(model, mask)
        # set_model_weights(model, model_, mask)

        save_path = os.path.join(train_args['save_path'], f"model_with_{percentage}_pct_stoch_run_{run_number}.pt")
        model = train(
            network=model,
            dataloader_train=dataloader,
            dataloader_val=dataloader_val,
            model_old=None,
            mask=mask,
            device=train_args['device'],
            epochs=train_args['epochs'],
            save_path=save_path
        )

        predictions_train = get_preds(model, dataloader)
        predictions_val = get_preds(model, dataloader_val)
        predictions_test = get_preds(model, dataloader_test)
        with open(
                os.path.join(
                    train_args['save_path'],
                    f"model_with_{percentage}_pct_stoch_run_{run_number}_preds.pkl"),
                'wb') as handle:
            pickle.dump({'train': predictions_train, 'val': predictions_val,'test': predictions_test},
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

