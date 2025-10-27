from .FFT_layer_linear import *
from .Spline_layer_linear import *

class KAN(nn.Module):
    def __init__(self, layer_dims, gridsize, layers_acts='F',addbias=True):
        '''
        Example:
            model = KAN([2,5,1], 12) 
            creates a model with two KAN layers:
                layer1 has 2 input nodes and 5 output nodes
                layer2 has 5 input nodes and 1 output node
                number of grids (frequencies for FFT) = 12
                only summation nodes.

            model = KAN([[2,0,0],[9,3,1],[1,0,0]], 12) 
            creates a model with two KAN layers:
                layer1 has 2 input nodes and 9 output nodes for addition
                    out of the 9 output nodes, right-most 6 nodes are used for 3 multiplications
                    then right-most 2 nodes are used for exponent.
                    So, if after summation, the nodes are 1,2,3,4,5,6,7,8,9, 
                        then final 5 output nodes of layer1 will be 1, 2^3 ,4*5 ,6*7, 8*9
                layer2 has 5 input nodes and 1 output node
                number of grids (frequencies for FFT) = 12
        '''
        super().__init__()
        assert isinstance(layer_dims, list), "`layer_dims` must be a list of ints or list of lists"
        assert isinstance(layers_acts, str), "`layers_acts` must be a string"
        if len(layers_acts) == 1:
            layers_acts = layers_acts * (len(layer_dims) - 1)
        if len(layers_acts) != len(layer_dims) - 1:
            raise ValueError("`layers_acts` length must be 1 or equal to number of layers")
        
        self.layers = nn.ModuleList()
        self.gridsize = gridsize
        self.mult_arity = 2        
        self.addbias = addbias
        
        input_dim, _, _ = self.set_layer_params(layer_dims[0])
        for item in layer_dims[1:]:
            out_dim, num_mult, num_exp = self.set_layer_params(item)
            if layers_acts[len(self.layers)] == 'S':
                self.layers.append(KAN_Spline_linear(
                    input_dim, out_dim, self.gridsize, 
                    num_mult, self.mult_arity, num_exp, 
                    self.addbias))
            else:
                self.layers.append(KAN_FFT_linear(
                    input_dim, out_dim, self.gridsize, 
                    num_mult, self.mult_arity, num_exp, 
                    self.addbias))
            input_dim = out_dim - num_mult * (self.mult_arity - 1) - num_exp
        # self.act = nn.ReLU()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x) 
            # if i < len(self.layers) - 1:  # skip ReLU after last layer
            #     x = self.act(x)
        return x
    
    def set_layer_params(self, param):
        num_mult=0
        num_exp=0
        if isinstance(param, int):
            out_dim = param
        elif isinstance(param, list):
            if len(param) >= 1:
                out_dim = param[0]
            if len(param) >= 2:
                num_mult = param[1]
            if len(param) >= 3:
                num_exp = param[2]
        else:
            print('invalid layer_dims to KAN_FFT ', param, ' pass either an int or a list of ints.')
        return out_dim, num_mult, num_exp
        

class KAN_FFT_naive(nn.Module):
    def __init__(self, layer_dims, gridsize, addbias=True):
        '''
        Example:
            model = KAN_FFT([2,5,1], 12) 
            creates a model with two KAN layers:
                layer1 has 2 input nodes and 5 output nodes
                layer2 has 5 input nodes and 1 output node
                number of grids (frequencies for FFT) = 12
                only summation nodes.

            model = KAN_FFT([[2,0,0],[9,3,1],[1,0,0]], 12) 
            creates a model with two KAN layers:
                layer1 has 2 input nodes and 9 output nodes for addition
                    out of the 9 output nodes, right-most 6 nodes are used for 3 multiplications
                    then right-most 2 nodes are used for exponent.
                    So, if after summation, the nodes are 1,2,3,4,5,6,7,8,9, 
                        then final 5 output nodes of layer1 will be 1, 2^3 ,4*5 ,6*7, 8*9
                layer2 has 5 input nodes and 1 output node
                number of grids (frequencies for FFT) = 12
        '''
        super().__init__()
        assert isinstance(layer_dims, list), "`layer_dims` must be a list of ints or list of lists"
        self.layers = nn.ModuleList()
        self.gridsize = gridsize
        self.mult_arity = 2        
        self.addbias = addbias
        
        input_dim, _, _ = self.set_layer_params(layer_dims[0])
        for item in layer_dims[1:]:
            out_dim, num_mult, num_exp = self.set_layer_params(item)
            self.layers.append(KAN_FFT_linear(
                    input_dim, out_dim, self.gridsize, 
                    num_mult, self.mult_arity, num_exp, 
                    self.addbias))
            input_dim = out_dim - num_mult * (self.mult_arity - 1) - num_exp
        # self.act = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x) 
            # if i < len(self.layers) - 1:  # skip ReLU after last layer
            #     x = self.act(x)
        return x
    
    def set_layer_params(self, param):
        num_mult=0
        num_exp=0
        if isinstance(param, int):
            out_dim = param
        elif isinstance(param, list):
            if len(param) >= 1:
                out_dim = param[0]
            if len(param) >= 2:
                num_mult = param[1]
            if len(param) >= 3:
                num_exp = param[2]
        else:
            print('invalid layer_dims to KAN_FFT ', param, ' pass either an int or a list of ints.')
        return out_dim, num_mult, num_exp
        