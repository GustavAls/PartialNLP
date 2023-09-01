import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models, losses
from torch.nn import MSELoss
from torch.optim import SGD
import torch.nn as nn
import os
from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score
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
    optimizer = SGD(network.parameters(), lr=0.001)

    loss_fn = MSELoss()
    best_score = 0
    best_model = None
    for epoch in trange(epochs, desc="Training MAP network"):
        network.train()
        for idx, (batch, label) in enumerate(dataloader_train):
            optimizer.zero_grad()
            batch = batch.to(device)
            label = label.to(device)
            prediction = network(batch)
            loss = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            network.eval()
            predictions = []
            labels = []
            for idx, (batch, label) in enumerate(dataloader_val):
                predictions.append(network(batch).detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)

            current_score = balanced_accuracy_score(labels.argmax(-1), predictions.argmax(-1))

            if current_score > best_score:
                best_score = current_score
                best_model = deepcopy(network)

    if best_model is None:
        UserWarning("The model failed to improve, something went wrong")
    else:
        torch.save(best_model.state_dict(), save_path)
        print(f"Model was saved to location {save_path}, terminated with score {best_score}")

    return best_model
