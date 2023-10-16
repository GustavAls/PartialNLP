import torch
import torch.nn as nn

TRANSFORMER_INCOMPATIBLE_MODULES = (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d,
                                    nn.BatchNorm2d, nn.BatchNorm3d)

TRANSFORMER_COMPATIBLE_MODULES = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.Conv1d)

def get_ignore_modules(self):
    ignore_modules = []
    for module in self.modules():
        if len([child for child in module.children()]) > 0:
            continue
        if not getattr(module, 'partial', False):
            ignore_modules.append(module)

    return ignore_modules

class PartialConstructor:
    def __init__(self, model, module_names):
        self.model = model
        self.module_names = module_names


    def select(self):
        counter = 0
        for name, module in self.model.named_modules():
            if name in self.module_names and isinstance(module, TRANSFORMER_COMPATIBLE_MODULES):
                setattr(module, 'partial', True)
                counter += 1
            else:
                setattr(module, 'partial', False)

        self.model.get_ignore_modules = get_ignore_modules
        setattr(self.model, 'num_partial_layers',  counter)
        return None

    def create_subnet_mask_list(self):
        subnet_mask_list = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0 or len(list(module.parameters())) == 0:
                continue
            if name in self.module_names:
                mask_method = torch.ones_like
            else:
                mask_method = torch.zeros_like
            subnet_mask_list.append(mask_method(nn.utils.parameters_to_vector(module.parameters())))
        subnet_mask = torch.cat(subnet_mask_list).bool()
        subnet_mask = subnet_mask.nonzero(as_tuple=True)[0]
        return subnet_mask








