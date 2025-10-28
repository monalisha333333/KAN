import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_Linear_SELU(nn.Module):
    def __init__(self, inputdim, outdim, 
                 num_mult=0, mult_arity=0, num_exp=0, addbias=True):
        super().__init__()
        self.inputdim = inputdim
        self.outdim = outdim
        self.num_mult = num_mult
        self.mult_arity = mult_arity
        self.num_exp = num_exp
        self.addbias = addbias

        # Simple linear layer
        self.linear = nn.Linear(inputdim, outdim, bias=addbias)

        # SELU activation for self-normalizing networks
        # self.activation = nn.SELU()
        self.activation = nn.ReLU()

    def forward(self, x):
        # Forward through linear + activation
        y = self.linear(x)
        y = self.activation(y)

        # Handle multiplicative and exponential terms (optional)
        if self.num_mult > 0 or self.num_exp > 0:
            total_mult_inputs = self.num_mult * self.mult_arity 
            total_exp_inputs = self.num_exp * 2 
            total_mult_exp_inputs = total_mult_inputs + total_exp_inputs

            assert self.outdim >= total_mult_exp_inputs, (
                "Not enough output dimensions to perform mult/exp operations"
            )

            keep_dim = self.outdim - total_mult_exp_inputs
            kept = y[:, :keep_dim]
            current = y[:, keep_dim:]

            # Multiplicative terms
            if self.num_mult > 0:
                mult_grouped = current[:, :total_mult_inputs].view(y.shape[0], self.num_mult, self.mult_arity)
                multed = torch.prod(mult_grouped, dim=-1)
            else:
                multed = torch.empty((y.shape[0], 0), device=y.device)

            # Exponential terms (x^y)
            if self.num_exp > 0:
                exp_section = current[:, total_mult_inputs:total_mult_inputs + total_exp_inputs]
                exp_grouped = exp_section.view(y.shape[0], self.num_exp, 2)
                # powered = torch.pow(exp_grouped[:, :, 0], exp_grouped[:, :, 1])
                base = torch.sigmoid(exp_grouped[:, :, 0])   # range in (0, 1), centered near 0.5
                exponent = torch.sigmoid(exp_grouped[:, :, 1])  # also in (0, 1)
                base = base * 0.9 + 0.05      # → (0.05, 0.95)
                exponent = exponent * 0.9 + 0.05
                powered = torch.pow(base, exponent)
            else:
                powered = torch.empty((y.shape[0], 0), device=y.device)

            y = torch.cat([kept, multed, powered], dim=-1)

        return y
