import torch.nn as nn
import torch
import numpy
from torch.distributions import Normal
from MAP_baseline.MapNN import MapNN


class BaseMAPLoss(nn.Module):

    def __init__(self, model, **kwargs):
        super(BaseMAPLoss, self).__init__()
        self.model = model
        self.likelihood = None
        self.prior = None

    def prior_probability(self):
        return self.prior.log_prob(self.parameter_vector())

    def forward(self, predictions, labels):
        return self.likelihood(predictions, labels) - self.prior_probability().mean()

    def parameter_vector(self):
        return nn.utils.parameters_to_vector(self.model.parameters())


class GLLGP_loss(BaseMAPLoss):
    """
    Loss for Gaussian Likelihood (GLL) with a Gaussian Prior (GP) on model weights
    """

    def __init__(self, prior_mu=0, prior_sigma=1, model=None, **kwargs):
        super().__init__(model)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.likelihood = nn.MSELoss()
        self.prior = Normal(self.prior_mu, self.prior_sigma)

class GLLCP_loss(BaseMAPLoss):
    """
    Loss for Gaussian Likelihood (GLL) with costum prior (CP)
    """

    def __init__(self, costum_prior, model, **kwargs):
        super(GLLCP_loss, self).__init__(model)
        if hasattr(costum_prior, 'log_prob'):
            self.costum_prior = costum_prior
        else:
            raise ValueError("Costum prior must be able to calculalte log_prob")

        self.likelihood = nn.MSELoss()

class CLLGP_loss(BaseMAPLoss):
    """
    Loss for classification with cross entropy likehood (CLL) and Gaussian Prior
    """

    def __init__(self, prior_mu, prior_sigma, model, weights=None, **kwargs):
        super(CLLGP_loss, self).__init__(model)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.likelihood = nn.CrossEntropyLoss(weight=weights)
        self.prior = Normal(self.prior_mu, self.prior_sigma)


class CLLCP_loss(BaseMAPLoss):
    """
    Loss for Classification (cross entropy) likelihood (GLL) with costum prior (CP)
    """

    def __init__(self, costum_prior, model, **kwargs):
        super(CLLCP_loss, self).__init__(model)
        if hasattr(costum_prior, 'log_prob'):
            self.costum_prior = costum_prior
        else:
            raise ValueError("Costum prior must be able to calculalte log_prob")

        self.likelihood = nn.CrossEntropyLoss()

