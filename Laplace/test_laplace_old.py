import Laplace
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_one = nn.Linear(10, 10)
        self.activation = nn.ReLU()
        self.out = nn.Linear(10, 1)
    def forward(self, x):
        out = self.layer_one(x)
        out = self.activation(out)
        out = self.out(out)
        return out

class MyDataset(Dataset):
    def __init__(self):
        self.X = np.random.normal(0,1, (100, 10))
        beta = np.random.normal(0,1, (10, 1))
        self.y = self.X @ beta + np.random.normal(0,0.1, (100, 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        x = torch.from_numpy(self.X[item])
        y = torch.from_numpy(self.y[item])

        return x, y


def train_laplace():
    model = MyModel()
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=10)
    la = Laplace(model, likelihood='regression', subset_of_weights='all', hessian_structure='kron')
    la.fit(dataloader)
    breakpoint()


if __name__ == '__main__':
    train_laplace()