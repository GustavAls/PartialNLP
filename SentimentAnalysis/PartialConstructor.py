import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader, Dataset
import numpy as np

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
    def __init__(self, model, module_names = None):
        self.model = model
        self.module_names = module_names
        if self.module_names == 'all':
            self.module_names = [name for name, _ in self.model.named_modules()]
        self.subnet_indices = None

    def select_random_percentile(self, num_params):
        counter = 0
        names = []
        for name, module in self.model.named_modules:
            if isinstance(module, TRANSFORMER_COMPATIBLE_MODULES):
                counter += 1
                names.append(name)
        self.module_names = np.random.choice(names, size=(num_params, ))

    def select(self):
        counter = 0
        for name, module in self.model.named_modules():
            if name in self.module_names and isinstance(module, TRANSFORMER_COMPATIBLE_MODULES):
                setattr(module, 'partial', True)
                counter += len(list(module.parameters()))
            else:
                setattr(module, 'partial', False)

        self.model.get_ignore_modules = get_ignore_modules
        setattr(self.model, 'num_partial_layers', counter)
        return None

    def get_subnetwork_indices(self):
        #  TODO implement subnetwork indices selection for partial within module work
        return self.subnet_indices

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


def mean_reduction(predictions):
    if predictions.ndim != 3:
        raise ValueError("Predictions should be on the form (batch_size, num_classes, num_mc_samples)")

    return predictions.mean(-1)


def median_reduction(predictions):
    if predictions.ndim != 3:
        raise ValueError("Predictions should be on the form (batch_size, num_classes, num_mc_samples)")

    return torch.median(predictions, -1)


def identity_reduction(predictions):
    return predictions


class PartialConstructorSwag(nn.Module):
    def __init__(self, model, n_iterations_between_snapshots=100, module_names=None, num_columns=0, num_mc_samples=50,
                 min_var=1e-20, reduction='mean', num_classes=2):
        super(PartialConstructorSwag, self).__init__()

        self.model = model
        self.device = next(model.parameters()).device
        self.number_of_iterations_bs = n_iterations_between_snapshots
        self.module_names = module_names
        self.num_columns = num_columns
        self.forward_call_iterator = 0
        self.has_called_select = False
        self.num_mc_samples = num_mc_samples
        self.min_var = min_var
        self.num_classes = num_classes

        if reduction == 'mean':
            self.reduction = mean_reduction

        elif reduction == 'median':
            self.reduction = median_reduction

        elif reduction == 'None':
            self.reduction = identity_reduction

        self.squared_theta_mean = None
        self.deviations = []
        self.n_parameters = 0
        self.n_snapshots = 0
        self.theta_mean = None
        self.swag_eval = False

    def parameter_to_vector(self):
        return nn.utils.parameters_to_vector((param for param in self.model.parameters() if param.requires_grad))

    def vector_to_parameters(self, param_new):
        return nn.utils.vector_to_parameters(param_new,
                                             (param for param in self.model.parameters() if param.requires_grad))

    def _init_parameters(self):

        if not self.has_called_select:
            raise ValueError("You must call select() before parameter initialisation")

        parameters_to_learn = self.parameter_to_vector()
        self.n_parameters = len(parameters_to_learn)
        self.theta_mean = torch.zeros_like(parameters_to_learn)
        self.squared_theta_mean = torch.zeros_like(parameters_to_learn)

    def select(self):
        for name, module in self.model.named_modules():
            if name in self.module_names:
                module.requires_grad_(True)
            else:
                module.requires_grad_(False)
        self.has_called_select = True
        self._init_parameters()

    def snapshot(self):

        new_parameter_values = self.parameter_to_vector()
        old_factor, new_factor = self.n_snapshots / (self.n_snapshots + 1), 1 / (self.n_snapshots + 1)
        self.theta_mean = self.theta_mean * old_factor + new_parameter_values * new_factor
        self.squared_theta_mean = self.squared_theta_mean * old_factor + new_parameter_values ** 2 * new_factor

        if self.num_columns > 0:
            deviation = new_parameter_values - self.theta_mean
            if len(self.deviations) > self.num_columns:
                self.deviations.pop(0)

            self.deviations.append(deviation.reshape(-1, 1))

    def scheduler(self):
        return (self.forward_call_iterator % self.number_of_iterations_bs == 0) and self.forward_call_iterator > 0

    def eval_swa(self):
        self.swag_eval = True
        self.deviations = torch.cat(self.deviations, dim=-1)
        self.squared_theta_mean = (self.squared_theta_mean - self.theta_mean ** 2).clamp(self.min_var)

        return self.train(False)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None
                ):

        if not self.swag_eval:
            self.forward_call_iterator += 1
            if self.scheduler():
                self.snapshot()

        out = self.model(input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions,
                         output_hidden_states, return_dict)
        return out

    def sample(self):

        z1 = torch.normal(mean=torch.zeros((self.theta_mean.numel())), std=1.0).to(self.device)
        z2 = torch.normal(mean=torch.zeros(self.num_columns), std=1.0).to(self.device)

        if self.num_columns > 0:
            theta = self.theta_mean + 2 ** -0.5 * (self.squared_theta_mean ** 0.5 * z1) + (
                    2 * (self.num_columns - 1)) ** -0.5 * (
                            self.deviations @ z2[:, None]).flatten()
        else:
            theta = self.theta_mean + self.squared_theta_mean * z1

        self.vector_to_parameters(theta)

    def predict_mc(self, **kwargs):

        batch_size = kwargs['input_ids'].shape[0]
        predictions = torch.zeros((batch_size, self.num_classes, self.num_mc_samples))

        for mc_sample in range(self.num_mc_samples):
            out = self.model(**kwargs)
            predictions[:, :, mc_sample] = out.logits.detach()

        predictions = self.reduction(predictions)
        return predictions



class Extension(nn.Module):
    """
    Class to prepare a huggingface model for Laplace transform
    """
    def __init__(self, model):
        super(Extension, self).__init__()
        self.model = model

    def forward(self, **kwargs):
        kwargs.pop('labels', None)
        output_dict = self.model(**kwargs)
        logits = output_dict['logits']
        return logits.to(torch.float32)


class Truncater(torch.nn.Module):

    """
    Class to truncate a distilbert model from Huggingface, to make it easier to test validity of methods with
    Not to be used in any practical applications
    """
    def __init__(self, model):
        super(Truncater, self).__init__()

        self.embeddings = model.distilbert.embeddings
        self.classifier = model.classifier
        self.config = model.config
        self.num_labels = model.num_labels
        self.another_linear = torch.nn.Linear(768, 20)
        self.another_linear_v2 = torch.nn.Linear(20, 20)
        self.classifier = torch.nn.Linear(20, 2)


    def forward(
            self,
            input_ids =  None,
            attention_mask = None,
            head_mask= None,
            inputs_embeds = None,
            labels= None,
            output_attentions= None,
            output_hidden_states = None,
            return_dict= None,
    ):
        embeddings = self.embeddings(input_ids, inputs_embeds)
        embeddings = torch.nn.ReLU()(self.another_linear(embeddings))
        embeddings = torch.nn.ReLU()(self.another_linear_v2(embeddings))
        logits = self.classifier(embeddings[:, 0])  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


class ForfunData(Dataset):

    def __init__(self):
        self.X = torch.randn(100, 10)
        y = torch.randint(0, 2, size=(100, 1))
        self.y = torch.zeros((100, 2))
        for i in range(y.shape[0]):
            self.y[y[i]] = 1

        self.return_normal = True

    def get_normal(self, item):
        return self.X[item], self.y[item]

    def get_unnormal(self, item):
        return {'x': self.X[item], 'labels': self.y[item]}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        if self.return_normal:
            return self.X[item], self.y[item].reshape(-1)
        else:
            return self.get_unnormal(item)