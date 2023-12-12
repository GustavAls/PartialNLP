import torch
from torch import optim
from typing import List
from torch import nn
import torch.nn.functional as F


def get_nonlinearity_from_string(non_linearity):
    if non_linearity == "leaky_relu":
        return nn.LeakyReLU()
    elif non_linearity == "tanh":
        return nn.Tanh()
    elif non_linearity == "relu":
        return nn.ReLU()
    elif non_linearity == "silu":
        return nn.SiLU()


class MapNNV2(nn.Module):
    def __init__(self, input_size: int, width: int, size: int, output_size: int = 1, non_linearity: str = "silu"):
        """
        Initialize network
        Args:
            input_size: size of input
        """
        # Superclass constructor
        super().__init__()
        self.non_linearity = get_nonlinearity_from_string(non_linearity)
        self.linear1 = nn.Linear(in_features=input_size, out_features=width)
        self.linear2 = nn.Linear(in_features=width, out_features=width)
        self.linear3 = nn.Linear(in_features=width, out_features=width)
        self.linear4 = nn.Linear(in_features=width, out_features=width)
        self.out = nn.Linear(in_features=width, out_features=output_size)


    def forward(self, x):
        """
        Forwards pass
        Args:
            x: Input vector.

        Returns:
        """
        output = self.linear1(x)
        output = self.non_linearity(output)
        output = self.linear2(output)
        output = self.non_linearity(output)
        output = self.linear3(output)
        output = self.non_linearity(output)
        output = self.linear4(output)
        output = self.non_linearity(output)
        output = self.out(output)
        return output


class MapNNRamping(nn.Module):

    def __init__(self, input_size, width, output_size, num_hidden_layers, non_linearity = 'silu'):
        super(MapNNRamping, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.width = width
        self.num_hidden_layers = num_hidden_layers
        self.non_linearity = get_nonlinearity_from_string(non_linearity)

        self.layers = [nn.Linear(self.input_size, self.width), nn.LeakyReLU()]

        for i in range(self.num_hidden_layers):
            self.layers.append(nn.Linear(self.width, self.width))
            self.layers.append(nn.LeakyReLU())
        self.layers += [nn.Linear(self.width, self.output_size)]

        self.layers = nn.Sequential(*self.layers)
    def forward(self, x):
        return self.layers(x)
class MapNN(nn.Module):
    """
    MAP neural network
    """
    def __init__(self, input_size: int, width: int, output_size: int = 1, non_linearity: str = "silu"):
        """
        Initialize network
        Args:
            input_size: size of input
        """
        # Superclass constructor
        super().__init__()
        self.output_size = output_size
        self.layers = nn.Sequential()
        self.non_linearity = get_nonlinearity_from_string(non_linearity)
        self.linear1 = nn.Linear(in_features=input_size, out_features=width)
        self.linear2 = nn.Linear(in_features=width, out_features=width)
        self.out = nn.Linear(in_features=width, out_features=output_size)


    def forward(self, x):
        """
        Forwards pass
        Args:
            x: Input vector.

        Returns:
        """
        output = self.linear1(x)
        output = self.non_linearity(output)
        output = self.linear2(output)
        output = self.non_linearity(output)
        output = self.out(output)
        return output


class MAPNNLike(nn.Module):

    def __init__(self, input_size, width, output_size):
        super(MAPNNLike, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layer_one = nn.Linear(in_features=self.input_size, out_features=width)
        self.layer_two = nn.Linear(in_features=width, out_features=width)
        self.out = nn.Linear(in_features=width, out_features=self.output_size)

        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.LeakyReLU()

    def forward(self, x):
        out = self.layer_one(x)
        out = self.activation1(out)
        out = self.layer_two(out)
        out = self.activation2(out)
        out = self.out(out)
        return out