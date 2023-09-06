import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import SGD, Adam
import torch.nn as nn
import os
from copy import deepcopy
from tqdm import trange


def train(network: nn.Module,
          dataloader_train,
          dataloader_val,
          device = 'cpu',
          epochs = 50,
          save_path = None):

    """

    :param network: (nn.Module) feed forward classification model
    :param dataloader_train: dataloader for the training cases, should output (inputs, labels)
    :param dataloader_val: dataloader for the validation cases, should output (inputs, labels)
    :return: The trained model (the best version of the trained model, from eval on validation set)
    """

    network.to(device)
    # optimizer = SGD(network.parameters(), lr=0.1)
    optimizer = Adam(network.parameters(), lr=0.01)

    loss_fn = MSELoss()
    best_loss = np.infty
    best_model = None
    for epoch in trange(epochs, desc="Training MAP network"):
        network.train()
        for idx, (batch, target) in enumerate(dataloader_train):
            optimizer.zero_grad()
            batch = batch.to(device)
            target = target.to(device)
            output = network(batch)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            network.eval()
            current_loss = 0
            for idx, (batch, target) in enumerate(dataloader_val):
                batch = batch.to(device)
                target = target.detach().cpu().numpy()
                output = network(batch).detach().cpu().numpy()
                current_loss += loss_fn(torch.FloatTensor(output), torch.FloatTensor(target))
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
