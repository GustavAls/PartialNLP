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


class MapNN(nn.Module):
    """
    MAP neural network
    """
    def __init__(self, input_size: int, width: int, size: int, output_size: int = 1, non_linearity: str = "silu"):
        """
        Initialize network
        Args:
            input_size: size of input
        """
        # Superclass constructor
        super().__init__()
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
