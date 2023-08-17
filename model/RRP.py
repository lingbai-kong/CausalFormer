import torch
import torch.nn as nn
import torch.nn.functional as F
from model.NonParamRP import *

class RegRelProp(RelProp):
    """
    This class implements the Regularized Relevance Propagation (RRP) algorithm.
    """
    def relprop(self, R):
        """
        This method performs the regression relevance propagation for the linear layer.

        Args:
            R (torch.Tensor): Relevance scores from the previous layer.

        Returns:
            R (torch.Tensor): Relevance scores for the current layer.
        """
        Z = F.linear(self.X, self.weight)
        S = safe_divide(R, Z)
        R = self.X * torch.autograd.grad(Z, self.X, S)[0]
        return R

class Linear(nn.Linear, RegRelProp):
    """
    This class extends the nn.Linear class to incorporate the regression Relevance Propagation (RRP) algorithm.
    It performs regression relevance propagation through linear layers.
    """
    pass
