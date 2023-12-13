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
        self.use_only_attn = False
        self.use_only_mlp = False

    def set_use_only_mlp(self):
        self.use_only_mlp = True
        self.use_only_attn = False

    def set_use_only_attn(self):
        self.use_only_mlp = False
        self.use_only_attn = True

    def named_modules(self):
        attn = 'attention'
        if self.use_only_mlp:
            restriction = lambda x, y: (not (attn in x)) and isinstance(y, TRANSFORMER_COMPATIBLE_MODULES)
        elif self.use_only_attn:
            restriction = lambda x, y: attn in x and isinstance(y, TRANSFORMER_COMPATIBLE_MODULES)
        else:
            restriction = lambda x,y: '' in x and isinstance(y, TRANSFORMER_COMPATIBLE_MODULES)

        return ((n, m) for n, m in self.model.named_modules() if restriction(n, m))

    def select_random_percentile(self, num_params):
        counter = 0
        names = []
        for name, module in self.named_modules():
            names.append(name)

        self.module_names = np.random.choice(names, size=(num_params, ))

    def select_max_operator_norm(self, num_params):

        names, norms = [], []
        for name, module in self.named_modules():
            names.append(name)
            norms.append(torch.linalg.matrix_norm(module.weight.data, ord = 2).item())

        argsorted = np.argsort(norms)
        self.module_names = [names[i] for i in argsorted[::-1][:num_params]]


    def select_min_operator_norm(self, num_params):
        names, norms = [], []
        for name, module in self.named_modules():
            names.append(name)
            norms.append(torch.linalg.matrix_norm(module.weight.data, ord=2).item())

        argsorted = np.argsort(norms)

        self.module_names = [names[i] for i in argsorted[:num_params]]
    def select_subnetwork_indices_module_wise(self, quantile = 1):

        for name, module in self.model.named_modules():
            if module.partial:
                parameter_vector = nn.utils.parameters_to_vector(module.parameters()).clone().detach()
                quant = torch.quantile(parameter_vector, torch.tensor(quantile))
                subnetwork_indices = torch.zeros_like(parameter_vector)
                subnetwork_indices[parameter_vector > quant] = 1
                subnetwork_indices = subnetwork_indices.unsqueeze(0).bool()
                setattr(module, 'subnetwork_indices', subnetwork_indices)


    def select_subnetwork_indices_for_kfac(self, percentile = 0.1):
        for name, module in self.model.named_modules():
            if module.partial:
                self.set_mask_for_kfac(module, percentile)

    def find_maximum_input_and_output_neurons(self, weight, percentile = 0.1):

        infty_norm = torch.linalg.norm(weight,ord = torch.inf,  dim = 0)
        highest_in = torch.argsort(infty_norm, descending=True)
        highest_in = highest_in[:int(percentile*len(highest_in))]
        highest_out = torch.argsort(torch.linalg.norm(weight[:, highest_in], ord = torch.inf, dim = 1), descending=True)
        highest_out = highest_out[:int(percentile*len(highest_out))]

        return highest_in, highest_out

    def select_all_modules(self):
        self.module_names = [name for name, module in self.named_modules()]

    def select_predifined_modules(self, module_names):
        self.module_names = module_names

    def select_sublayer_kfac(self,percentile = 0.1):
        for name, module in self.named_modules():
            if name in self.module_names:
                self.set_mask_for_sublayer_kfac(module, percentile)

    def set_mask_for_sublayer_kfac(self, module, percentile):

        param_indices, subnetwork_indices = [], []

        for name, param in module.named_parameters():
            if name == 'weight':
                data = param.data.clone()
                highest_in, highest_out = self.find_maximum_input_and_output_neurons(data)
                param_indices.append([highest_in, highest_out])

                subnet_ind_in = torch.zeros_like(data)
                subnet_ind_out = torch.zeros_like(data)
                subnet_ind_in[:,highest_in] = 1
                subnet_ind_out[highest_out,:] = 1
                subnet_ind = subnet_ind_out * subnet_ind_in
                subnetwork_indices.append(subnet_ind.bool().flatten())

            elif name == 'bias':
                indices = torch.argsort(torch.abs(param))[:int(percentile * len(param))]
                param_indices.append(indices)
                subnet_ind = torch.zeros_like(param).bool()
                subnet_ind[indices] = True
                subnetwork_indices.append(subnet_ind)

        subnetwork_indices = torch.cat(subnetwork_indices, 0).unsqueeze(0)
        setattr(module, 'subnetwork_indices', subnetwork_indices)
        setattr(module, 'param_indices', param_indices)
        setattr(module, 'subsample_fisher_A_B', True)

    def set_mask_for_kfac(self, module, percentile = 0.1):

        if not isinstance(module, nn.Linear):
            raise ValueError("Module should be a nn.Linear")
        param_indices = []
        subnetwork_indices = []
        for name, param in module.named_parameters():
            if name == 'weight':
                data = param.data.clone()
                euclidian_norm = torch.linalg.norm(data, dim = 1)
                highest = torch.argsort(euclidian_norm, descending=True)
                num_to_be_chosen = int(len(euclidian_norm)*percentile)
                param_indices.append(highest[:num_to_be_chosen])
                subnet_ind = torch.zeros_like(data)
                subnet_ind[highest[:num_to_be_chosen]] = 1
                subnetwork_indices.append(subnet_ind.bool().flatten())
            elif name == 'bias':
                indices = torch.argsort(torch.abs(param))[:int(percentile * len(param))]
                param_indices.append(indices)
                subnet_ind = torch.zeros_like(param).bool()
                subnet_ind[indices] = True
                subnetwork_indices.append(subnet_ind)

        subnetwork_indices = torch.cat(subnetwork_indices, 0).unsqueeze(0)
        setattr(module, 'subnetwork_indices', subnetwork_indices)
        setattr(module, 'param_indices', param_indices)
        setattr(module, 'subsample_fisher_A_B', True)

    def get_params_from_module(self, module):
        pass

    def select(self):
        counter = 0
        for name, module in self.model.named_modules():
            if name in self.module_names and isinstance(module, TRANSFORMER_COMPATIBLE_MODULES):
                setattr(module, 'partial', True)
                module.requires_grad_(True)
                counter += len(list(module.parameters()))
            else:
                setattr(module, 'partial', False)
                module.requires_grad_(False)

        self.model.get_ignore_modules = get_ignore_modules
        setattr(self.model, 'num_partial_layers', counter)
        return None

    def select_last_layer(self):
        self.module_names = ['model.classifier']

    def get_num_stochastic_parameters(self):
        return len(nn.utils.parameters_to_vector(param for param in self.model.parameters() if param.requires_grad))

    def get_num_params(self):
        return len(nn.utils.parameters_to_vector(self.model.parameters()))
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


class PartialConstructorSwag:
    def __init__(self, model, n_iterations_between_snapshots=100, module_names=None, num_columns=0, num_mc_samples=50,
                 min_var=1e-20, reduction='mean', num_classes=2,use_sublayer = False, **kwargs):

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
        self.is_training = True
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
        self.step_counter = 0
        self.use_only_mlp = False
        self.use_only_attn = False
        self.use_sublayer = use_sublayer
        self.param_mask = None
        self.original_values = None

    def train(self, is_training = True):
        self.is_training = is_training
    def eval(self):
        return self.eval_swa()
    def parameters(self):
        return (param for param in self.model.parameters() if param.requires_grad)
    def parameter_to_vector(self):
        return nn.utils.parameters_to_vector((param for param in self.model.parameters() if param.requires_grad))

    def vector_to_parameters(self, param_new):
        return nn.utils.vector_to_parameters(param_new,
                                             (param for param in self.model.parameters() if param.requires_grad))

    def set_use_only_mlp(self):
        self.use_only_mlp = True
        self.use_only_attn = False

    def set_use_only_attn(self):
        self.use_only_mlp = False
        self.use_only_attn = True

    def named_modules(self):

        attn = 'attention'
        if self.use_only_mlp:
            restriction = lambda x, y: (not (attn in x)) and isinstance(y, TRANSFORMER_COMPATIBLE_MODULES)
        elif self.use_only_attn:
            restriction = lambda x, y: attn in x and isinstance(y, TRANSFORMER_COMPATIBLE_MODULES)
        else:
            restriction = lambda x, y: '' in x and isinstance(y, TRANSFORMER_COMPATIBLE_MODULES)

        return ((n, m) for n, m in self.model.named_modules() if restriction(n, m))

    def select_random_percentile(self, num_params):
        names = []
        for name, module in self.named_modules():
            names.append(name)
        self.module_names = np.random.choice(names, size = (num_params, ))

    def select_max_operator_norm(self, num_params):
        names, norms = [], []
        for name, module in self.named_modules():
            names.append(name)
            norms.append(torch.linalg.matrix_norm(module.weight.data, ord = 2).item())

        argsorted = np.argsort(norms)
        self.module_names = [names[i] for i in argsorted[::-1][:num_params]]

    def select_min_operator_norm(self, num_params):
        names, norms = [], []
        for name, module in self.named_modules():
            names.append(name)
            norms.append(torch.linalg.matrix_norm(module.weight.data, ord=2).item())

        argsorted = np.argsort(norms)

        self.module_names = [names[i] for i in argsorted[:num_params]]

    def select_max_l1_norm(self, percentile = 0.1):

        if self.module_names is None or len(self.module_names) == 0:
            raise ValueError("Must select module names before calling this function")

        self.use_sublayer = True
        param_mask = [], original_values = []
        for param in self.model.parameters():
            if param.requires_grad:
                vector_ = param.view(-1).clone().detach().cpu().numpy()
                percentile = np.percentile(vector_, q = 100-percentile)
                vector_ = vector_ > percentile
                param_mask.append(torch.from_numpy(vector_))
                original_values.append(param.view(-1).detach().clone())
        param_mask = torch.cat(param_mask, dim = 0).to(self.device)
        self.param_mask = param_mask
        self.original_values = torch.cat(original_values, dim = 0).to(self.device)

    def select_all_modules(self):
        self.module_names = [n for n, m in self.named_modules()]

    def _init_parameters(self):

        if not self.has_called_select:
            raise ValueError("You must call select() before parameter initialisation")

        parameters_to_learn = self.parameter_to_vector() if not self.use_sublayer else self.param_mask
        self.n_parameters = len(parameters_to_learn)

        self.theta_mean = torch.zeros_like(parameters_to_learn).to(self.device)
        self.squared_theta_mean = torch.zeros_like(parameters_to_learn).to(self.device)

    def init_new_model_for_optim(self, model):
        self.model = model
        self.select()
        self._init_parameters()
        self.step_counter = 0
        self.n_snapshots = 0
        self.train()
        self.deviations = []

    def select(self, percentile = 10):
        for name, module in self.model.named_modules():
            if name in self.module_names:
                module.requires_grad_(True)
            else:
                module.requires_grad_(False)
        self.has_called_select = True

        if self.use_sublayer:
            self.select_max_l1_norm(percentile=percentile)

        self._init_parameters()

    def snapshot(self):

        new_parameter_values = self.parameter_to_vector()
        old_factor, new_factor = self.n_snapshots / (self.n_snapshots + 1), 1 / (self.n_snapshots + 1)
        self.theta_mean = self.theta_mean * old_factor + new_parameter_values * new_factor
        self.squared_theta_mean = self.squared_theta_mean * old_factor + new_parameter_values ** 2 * new_factor

        self.n_snapshots += 1
        if self.num_columns > 0:
            deviation = new_parameter_values - self.theta_mean
            if len(self.deviations) > self.num_columns:
                self.deviations.pop(0)

            self.deviations.append(deviation.reshape(-1, 1))

    def apply_sublayer_mask(self, param_vect):
        if self.use_sublayer:
            if self.param_mask is not None:
                return param_vect[self.param_mask]
            else:
                raise ValueError("You must generate a param mask before applying the mask")
        return param_vect

    def scheduler(self):
        return (self.step_counter+1) % self.number_of_iterations_bs == 0

    def eval_swa(self):
        self.swag_eval = True
        self.deviations = torch.cat(self.deviations, dim=-1)
        self.squared_theta_mean = (self.squared_theta_mean - self.theta_mean ** 2).clamp(self.min_var)
        self.model.eval()
        return self.train(False)

    def reset_params_with_mask(self):
        if self.use_sublayer:
            vect = self.parameter_to_vector()
            vect[~self.param_mask] = self.original_values[~self.param_mask]
            self.vector_to_parameters(vect)

    def __call__(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None
                ):


        if self.is_training:
            out = self.model(input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions,
                         output_hidden_states, return_dict)

            self.step_counter += 1

        else:
            kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask,
                      'input_embeds': inputs_embeds, 'labels': labels, 'output_attentions': output_attentions,
                      'output_hidden_states': output_hidden_states, 'return_dict': return_dict}
            out = self.predict_mc(**kwargs)
        return out

    def sample(self):

        num_columns = self.deviations.shape[-1]
        z1 = torch.normal(mean=torch.zeros((self.theta_mean.numel())), std=1.0).to(self.device)
        z2 = torch.normal(mean=torch.zeros(num_columns), std=1.0).to(self.device)

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

        softmax = nn.Softmax(dim = 1)
        with torch.no_grad():
            for mc_sample in range(self.num_mc_samples):
                self.sample()
                out = self.model(**kwargs)
                predictions[:, :, mc_sample] = out.logits.detach().cpu()
            predictions = softmax(predictions)
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