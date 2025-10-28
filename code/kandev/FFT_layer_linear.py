import torch
import torch.nn as nn
import numpy as np

class KAN_FFT_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, num_mult=0, mult_arity=0, num_exp=0, addbias=True):
        super().__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.num_mult = num_mult
        self.mult_arity = mult_arity
        self.num_exp = num_exp

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self,x):
        # print('device x', x.device)
        # print('self.fouriercoeffs device',self.fouriercoeffs.device)
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))
        # print('k device', k.device)
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)

        if self.num_mult > 0 or self.num_exp > 0:
            total_mult_inputs = self.num_mult * self.mult_arity 
            total_exp_inputs = self.num_exp * 2 
            total_mult_exp_inputs = total_mult_inputs + total_exp_inputs
            assert self.outdim >= total_mult_exp_inputs, "Not enough output dimensions to perform mult/exp operations"
        
            keep_dim = self.outdim - total_mult_exp_inputs
            kept = y[:, :keep_dim]  # First few nodes unchanged
            current = y[:, keep_dim:]  # Remaining part for mult & exp
            if self.num_mult > 0:
                mult_grouped = current[:, :total_mult_inputs].view(y.shape[0], self.num_mult, self.mult_arity)  # (batch_size, num_mult, arity)
                multed = torch.prod(mult_grouped, dim=-1)  # Multiply along arity dim
            else:
                multed = torch.empty((y.shape[0], 0), device=y.device)
            if self.num_exp > 0:
                exp_section = current[:, total_mult_inputs:total_mult_inputs + total_exp_inputs]
                exp_grouped = exp_section.view(y.shape[0], self.num_exp, 2)  # (batch, num, 2)
                base = torch.sigmoid(exp_grouped[:, :, 0])   # range in (0, 1), centered near 0.5
                exponent = torch.sigmoid(exp_grouped[:, :, 1])  # also in (0, 1)
                base = base * 0.9 + 0.05      # → (0.05, 0.95)
                exponent = exponent * 0.9 + 0.05
                powered = torch.pow(base, exponent)
              # powered = torch.pow(exp_grouped[:, :, 0], exp_grouped[:, :, 1])
            else:
                powered = torch.empty((y.shape[0], 0), device=y.device)
            
            y = torch.cat([kept, multed, powered], dim=-1)
        return y
    
class KAN_FFT_linear_w_base(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, 
                 num_mult=0, mult_arity=0, num_exp=0, addbias=True,
                 scale_base_mu=0.0, 
                 scale_base_sigma=1.0, scale_fft=1.0, fft_trainable=True,
                 base_fun=torch.nn.SiLU()):
        super().__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.num_mult = num_mult
        self.mult_arity = mult_arity
        self.num_exp = num_exp

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

        self.mask = torch.nn.Parameter(torch.ones(inputdim, outdim)).requires_grad_(False)
        
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(inputdim) + \
                         scale_base_sigma * (torch.rand(inputdim, outdim)*2-1) * 1/np.sqrt(inputdim)).requires_grad_(fft_trainable)
        self.scale_fft = torch.nn.Parameter(torch.ones(inputdim, outdim) * scale_fft * 1 / np.sqrt(inputdim) * self.mask).requires_grad_(fft_trainable)  # make scale trainable
        self.base_fun = base_fun

    def forward(self,x):
        # print('device x', x.device)
        # print('self.fouriercoeffs device',self.fouriercoeffs.device)
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bij", torch.concat([c, s], axis=0), self.fouriercoeffs)
       
        base = self.base_fun(x) # (batch, in_dim)
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_fft[None,:,:] * y
        y = self.mask[None,:,:] * y
            
        y = torch.sum(y, dim=1)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)

        if self.num_mult > 0 or self.num_exp > 0:
            total_mult_inputs = self.num_mult * self.mult_arity 
            total_exp_inputs = self.num_exp * 2 
            total_mult_exp_inputs = total_mult_inputs + total_exp_inputs
            assert self.outdim >= total_mult_exp_inputs, "Not enough output dimensions to perform mult/exp operations"
        
            keep_dim = self.outdim - total_mult_exp_inputs
            kept = y[:, :keep_dim]  # First few nodes unchanged
            current = y[:, keep_dim:]  # Remaining part for mult & exp
            if self.num_mult > 0:
                mult_grouped = current[:, :total_mult_inputs].view(y.shape[0], self.num_mult, self.mult_arity)  # (batch_size, num_mult, arity)
                multed = torch.prod(mult_grouped, dim=-1)  # Multiply along arity dim
            else:
                multed = torch.empty((y.shape[0], 0), device=y.device)
            if self.num_exp > 0:
                exp_section = current[:, total_mult_inputs:total_mult_inputs + total_exp_inputs]
                exp_grouped = exp_section.view(y.shape[0], self.num_exp, 2)  # (batch, num, 2)
                base = torch.sigmoid(exp_grouped[:, :, 0])   # range in (0, 1), centered near 0.5
                exponent = torch.sigmoid(exp_grouped[:, :, 1])  # also in (0, 1)
                base = base * 0.9 + 0.05      # → (0.05, 0.95)
                exponent = exponent * 0.9 + 0.05
                powered = torch.pow(base, exponent)

                # powered = torch.pow(exp_grouped[:, :, 0], exp_grouped[:, :, 1])
            else:
                powered = torch.empty((y.shape[0], 0), device=y.device)
            
            y = torch.cat([kept, multed, powered], dim=-1)
        return y