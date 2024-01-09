
import numpy as np
import torch.nn as nn
import torch
import pickle
from tqdm import tqdm

class TemperatureScaler(nn.Module):

    def __init__(self, predictions, labels):
        super(TemperatureScaler, self).__init__()

        self.predictions = predictions
        self.labels = labels

        self.temperature = nn.Parameter(torch.ones(1)*1.2)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD([self.temperature], lr = 0.01)

    def forward(self, x):
        temp = self.temperature.unsqueeze(1).expand(1, x.shape[-1])
        out = x / temp
        return out

    def optimize(self, num_iterations):
        pbar = tqdm(range(num_iterations), desc = 'Optimizing temperature scaling')
        for niter in pbar:
            self.optimizer.zero_grad()
            out = self.forward(self.predictions)
            loss = self.loss_function(out, self.labels)
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix({'NLL': loss.item()})



