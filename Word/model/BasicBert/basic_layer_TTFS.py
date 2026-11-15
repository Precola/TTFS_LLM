import torch
import torch.nn as nn
from .base_TTFS import  TTFS_SNN_norm
from torch.nn import functional as F



'''
layer normalization
'''
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias, X_n=1000, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.B_n = X_n #(1 + 0.5) * X_n
        self.threshold = None
        self.D_i = nn.Parameter(torch.zeros(ndim,))
        # 256 is the hidden size of layernorm, it's independent of tokens.
        self.model = TTFS_SNN_norm(dim_in=ndim, dim_out=1, X_n=1000)
        # print() #Fine

    def _set_init_params(self, t_min_prev, t_min):

        # t_min, t_max = 0, 1.5  # for the input layer
        t_min_m1, t_max_m1, t_min_m2, t_max_m2 = self.model.layers[0].set_params(t_min_prev, t_min)  # mean
        t_min_sub1, t_max_sub1, t_min_sub2, t_max_sub2 = self.model.layers[1].set_params(t_min_m2, t_max_m2)  # sub
        t_min_sq1, t_max_sq1, t_min_sq2, t_max_sq2 = self.model.layers[2].set_params(t_min_sub2, t_max_sub2)  # sqr
        t_min_var, t_max_var = self.model.layers[3].set_params(t_min_sq2, t_max_sq2)
        #  self.model.layers[4] and self.model.layers[5] refer to [self.ttfs_root1] + [self.ttfs_root2], respectively. They have
        #  the same t_max and t_min...
        for layer in self.model.layers[4].layers:
            t_min_r1, t_max_r1 = layer.set_params(t_min_var, t_max_var)
        t_min_r2, t_max_r2 = self.model.layers[4].out.set_params(t_min_r1, t_max_r1)
        for layer in self.model.layers[5].layers:
            t_min_r1, t_max_r1 = layer.set_params(t_min_var, t_max_var)
        self.model.layers[5].out.set_params(t_min_r1, t_max_r1)
        for layer in self.model.layers[6].layers:
            t_min_r1, t_max_r1 = layer.set_params(t_min_var, t_max_var)
        self.model.layers[6].out.set_params(t_min_r1, t_max_r1)

        t_min_norm, t_max_norm = self.model.set_params(t_min_r2, t_max_r2)

        # t_min_norm, t_max_norm = self.model.set_params(t_min_var, t_max_var)

        return t_min_norm, t_max_norm

    def set_params(self, t_min_prev, t_min):
        t_min_norm, t_max_norm = self._set_init_params(t_min_prev, t_min)

        self.t_min_prev = t_min_norm
        self.t_min = t_max_norm
        self.t_max =t_max_norm + self.B_n
        return self.t_min, self.t_max

    def forward(self, t_j, delta=False):
        """
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer.
        """
        self.threshold = self.t_max - self.t_min - self.D_i
        #print(self.D_i.min(), self.D_i.max())
        # z =  self.linear(t_j - self.t_min) + self.threshold + self.t_min
        # print(f'______________layerNorm:{t_j.max(),t_j.min()}________________')
        x_norm = self.model(t_j)

        z =  self.weight * (x_norm - self.t_min) + self.threshold + self.t_min
        # z =  self.weight * (t_j - self.t_min) + self.threshold + self.t_min
        t_i = torch.where(z < self.t_max + self.B_n, z, self.t_max + self.B_n)

        if self.bias is not None:
            t_i += self.bias

        if delta == True:
            delta_t = t_i - self.t_max  # TODO !!
            return delta_t
        else:
            return t_i


class Atten_QK(nn.Module):
    '''
    voltage
    '''

    def __init__(self, dim_in, dim_out, bias= False, X_n=1000):
        super().__init__()
        self.t_min_prev, self.t_min_1, self.t_max_1 = 0, 0, 1
        self.B_n = X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        # self.threshold = None
        self.alpha = torch.ones((dim_out,))
        self.D_i = nn.Parameter(torch.zeros(dim_out, )) if bias else None
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min_1 = t_min
        self.t_max_1 = self.t_min_1 + self.B_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        return self.t_min_1, self.t_max_1, self.t_min_2, self.t_max_2

    def forward(self, t_j):
        """
        Input spiking times tj, output the value of membrane potential in case of output layer.
        """

        z = self.linear(t_j - self.t_min_1)
        self.alpha = self.D_i / (self.t_min_2 - self.t_min_prev)
        output = self.alpha * (self.t_min_2 - self.t_min_prev) + z

        return output


class Atten_V(nn.Module):
    '''
    spike time
    '''

    def __init__(self, dim_in, dim_out, bias= False, X_n=1000):
        super().__init__()
        self.t_min_prev, self.t_min_1, self.t_max_1 = 0, 0, 1
        self.B_n = X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None
        self.alpha = torch.ones((dim_out,))
        self.D_i = nn.Parameter(torch.zeros(dim_out, )) if bias else None
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min_1 = t_min
        self.t_max_1 = self.t_min_1 + self.B_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        return self.t_min_1, self.t_max_1, self.t_min_2, self.t_max_2

    def forward(self, t_j):
        """
        Input spiking times tj, output spiking times ti in case of output layer.
        """
        self.threshold = self.t_max_2 - self.t_min_1 - self.D_i
        # print(f'LN: {(t_j - self.t_min_1).min(), (t_j - self.t_min_1).max()}-------------')

        z = self.linear(t_j - self.t_min_1) + self.t_max_2 + self.D_i
        t_i = torch.where(z < self.t_max_2 + self.B_n, z, self.t_max_2 + self.B_n)

        return t_i


class Atten_QKV(nn.Module):
    def __init__(self, X_n=1000):
        super().__init__()
        self.t_min_prev, self.t_min_1, self.t_max_1 = 0, 0, 1
        self.B_n = X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min_1 = t_min
        self.t_max_1 = self.t_min_1 + self.B_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        return self.t_min_1, self.t_max_1, self.t_min_2, self.t_max_2

    def forward(self, t_in, weights):
        """
        input: spike time t_in
               membrane potential weights
        output: spike time
        """
        # print(f'v:{(t_in - self.t_min_1).min(), (t_in - self.t_min_1).max()},--------')
        z = weights @ (t_in - self.t_min_1) + self.t_max_2
        t_i = torch.where(z < self.t_max_2 + self.B_n, z, self.t_max_2 + self.B_n)

        return t_i

class TTFS_unit_gelu(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1000):
        super().__init__()

        self.layer = TTFS_unit_gen(dim_in, dim_out)  #TODO Gelu
        self.gelu = nn.GELU()

    def set_params(self, t_min_prev, t_min):
        t_tuple = self.layer.set_params(t_min_prev, t_min)
        t_min_prev, t_min = t_tuple[-2:]

        return t_min_prev, t_min

    def forward(self, x):
        x = self.layer(x)
        # print('x:', (x - self.layers[0].t_max_2).min(), (x - self.layers[0].t_max_2).max())
        x = self.gelu(x - self.layer.t_max_2) + self.layer.t_max_2
        return x


class TTFS_unit_relu(nn.Module):
    '''
    spike time
    input: two time bins
    output: one time bin
    '''

    def __init__(self, dim_in, dim_out, X_n=1000):
        super().__init__()
        self.t_min_prev, self.t_min_1, self.t_max_1 = 0, 0, 1
        self.B_n = X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None
        self.D_i = nn.Parameter(torch.zeros(dim_out, ))
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min_1 = t_min
        self.t_max_1 = self.t_min_1 + self.B_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        return self.t_min_1, self.t_max_1, self.t_min_2, self.t_max_2

    def forward(self, t_j):
        """
        Input spiking times tj, output spiking times ti in case of output layer.
        """
        self.threshold = self.t_max_2 - self.t_min_1 - self.D_i

        z = self.linear(t_j - self.t_min_1) + self.threshold + self.t_min_1
        t_i = torch.where(z < self.t_max_2, z, self.t_max_2)  # Relu

        return t_i


class TTFS_unit_gen(nn.Module):
    '''
    spike time
    input: two time bins
    output: two time bin
    '''

    def __init__(self, dim_in, dim_out, X_n=1000, bias=True):
        super().__init__()
        self.t_min_prev, self.t_min_1, self.t_max_1 = 0, 0, 1
        self.B_n = X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None
        self.D_i = nn.Parameter(torch.zeros(dim_out, )) if bias else None
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min_1 = t_min
        self.t_max_1 = self.t_min_1 + self.B_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        return self.t_min_1, self.t_max_1, self.t_min_2, self.t_max_2

    def forward(self, t_j):
        """
        Input spiking times tj, output spiking times ti in case of output layer.
        """

        self.threshold = self.t_max_2 - self.t_min_1 + self.D_i

        z = self.linear(t_j - self.t_min_1) + self.threshold + self.t_min_1
        t_i = torch.where(z < self.t_max_2 + self.B_n, z, self.t_max_2 + self.B_n)  # Relu

        return t_i


class TTFS_unit(nn.Module):
    '''
    spike time
    input: one time bins
    output: two time bin
    '''

    def __init__(self, dim_in, dim_out, X_n=1000):
        super().__init__()
        self.t_min_prev, self.t_min, self.t_max = 0, 0, 1
        self.B_n = X_n
        self.threshold = None
        self.D_i = nn.Parameter(torch.zeros(dim_out, ))
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min = t_min
        self.t_max = self.t_min + self.B_n
        return self.t_min, self.t_max

    def forward(self, t_j):
        """
        Input spiking times tj, output spiking times ti in case of output layer.
        """
        self.threshold = self.t_max - self.t_min - self.D_i

        z = self.linear(t_j - self.t_min) + self.threshold + self.t_min
        t_i = torch.where(z < self.t_max + self.B_n, z, self.t_max + self.B_n)

        return t_i


class TTFS_dropout(nn.Module):
    '''
    spike time
    input: two time bins
    output: two time bin
    '''

    def __init__(self, p, X_n=1000):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.t_min_prev, self.t_min_1, self.t_max_1 = 0, 0, 1
        self.B_n = X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None
        self.p = p

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min_1 = t_min
        self.t_max_1 = self.t_min_1 + self.B_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        return self.t_min_1, self.t_max_1, self.t_min_2, self.t_max_2

    def forward(self, t_j):
        """
        Input spiking times tj, output spiking times ti in case of output layer.
        """
        if self.training:
            weight = (torch.rand_like(t_j) > self.p).float() / (1 - self.p)
            z = weight * (t_j - self.t_min_1) + self.t_max_2
        else:
            z = (t_j - self.t_min_1) + self.t_max_2
        t_i = torch.where(z < self.t_max_2 + self.B_n, z, self.t_max_2 + self.B_n)  # Relu

        return t_i
