import torch
import torch.nn as nn
import numpy


def _parameter_vector(model: nn.Module):
    return nn.utils.parameters_to_vector(model.parameters()).detach()

def set_gradient_parameters(model: nn.Module, module_dict: dict):
    """

    :param model: Fully trained model
    :param module_dict: (key, value) = (str, [str,..]) where the key is the module, and the list contains the names of
    the submodules, if the list is empty, the entire module is set to not require a gradient, else each subcomponent
    :return: model
    """

    model.requires_grad_(False)
    for key, val in module_dict.items():
        if len(val) == 0:
            model.get_submodule(key).requires_grad_(True)
        else:
            for v in val:
                model.get_submodule(key).get_submodule(v).requires_grad_(True)

    return model


