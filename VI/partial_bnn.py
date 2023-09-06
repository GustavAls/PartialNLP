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
import partial_bnn_functional as funct

class TestModelPartBNN(nn.Module):
    def __init__(self, bnn_part):
        super(TestModelPartBNN, self).__init__()

        self.module_one = bnn_part
        self.module_two = nn.Linear()
        self.activation = nn.ReLU()
        self.last_layer = nn.Linear(10, 2)
    def forward(self, x):
        out = self.module_one(x)
        out = self.activation(out)
        out = self.module_two(out)
        out = self.activation(out)
        out = self.last_layer(out)

        return out


class MiniTestModel(nn.Module):
    def __init__(self, bnn):
        super(MiniTestModel, self).__init__()

        self.bnn = bnn
        self.layer_one = nn.Linear(5, 1)
    def forward(self, x):
        inputs = torch.split(x, split_size_or_sections=5, dim = -1)
        upper_path = self.bnn(inputs[0])
        lower_part = self.layer_one(inputs[1])
        return upper_path + lower_part

class MiniBNNModel(nn.Module):
    def __init__(self):
        super(MiniBNNModel, self).__init__()
        self.layer = nn.Linear(5,5, bias=False)
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(5,1)
    def forward(self, x):
        out = self.layer(x)
        out = self.activation(out)
        out = self.layer1(out)
        return out
def bnn(model, mask = None):
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
def vector_mask_to_parameter_mask(vec, parameters) -> None:
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

    bnn_length = int(len(argsorted)*percentile/100)
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
            if module.bias:
                name_to_mask[name] = [param_mask[counter], param_mask[counter + 1]]
                counter += 2
            else:
                name_to_mask[name] = [param_mask[counter]]
                counter += 1
    return name_to_mask
class DataForFun(Dataset):
    def __init__(self):

        alpha = np.random.normal(0,1, (5, 1))
        beta = np.random.normal(10, 1, (1,))
        self.X = np.random.uniform(0, 10, size = (100, 5))
        self.y = self.X@alpha + beta

    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        inp = torch.from_numpy(self.X[item]).float()
        label = torch.from_numpy(self.y[item]).float()
        return inp, label


def set_variance_param_to_zero(model):

    for idx, p in enumerate(model.parameters()):
        if (idx+1) % 2 == 0:
            p.requires_grad_(False)
            p *= 0


if __name__ == '__main__':



    model = MiniBNNModel()
    # param_mask = create_mask(model, 50)
    # model_ = deepcopy(model)
    # bnn(model, mask=param_mask)
    # funct.set_model_weights(model, model_, param_mask)


    dataloader = DataLoader(DataForFun(), batch_size=10)
    dataloader_val = DataLoader(DataForFun(), batch_size=10)
    train_args = {'epochs': 10,
                  'device': 'cpu',
                  'save_path': r'C:\Users\45292\Documents\Master\VI Simple\Models\Test Cases'}
    funct.train_model_with_varying_stochasticity(
        model,dataloader,dataloader_val,range(10,100),train_args
    )
    # funct.train(model, dataloader,dataloader_val,model_old=None, mask=None, vi = False)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)



    for epoch in tqdm(range(10)):
        for x, label in dataloader:
            optimizer.zero_grad()
            output = model(x)
            kl = get_kl_loss(model)
            ce_loss = criterion(output, label)
            loss = ce_loss + kl / 10
            loss.backward()
            optimizer.step()
            funct.set_model_weights(model, model_, param_mask)


    breakpoint()