from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.nn.utils import skip_init
import numpy as np
import matplotlib.pyplot as plt

'''
tanh
'''
class TTFS_block_tanh(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=500,):
        super().__init__()
        self.t_min_prev, self.t_min_1, self.t_max_1=0, 0, 1
        self.B_n = X_n #(1 + 0.5) * X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        self.t = 1

        self.D_i = nn.Parameter(torch.zeros(dim_out,))
        # self.weight = nn.Parameter(torch.ones((dim_in, dim_out)) , requires_grad=False)
        # self.register_buffer("weight", torch.ones((dim_in, dim_out)))
        self.LN = nn.LayerNorm(dim_out, bias=False, eps=1e-12)

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
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer.
        """
        self.threshold = self.t_max_2 - self.t_min_1 - self.D_i
        z =  self.LN(self.linear(t_j))  - self.D_i #- self.threshold - self.t_min_2# - self.t_max_2 + self.t_max_2
        t_i = torch.clamp(z, min=-self.t, max=self.t)

        return t_i

'''
Sum
'''
class TTFS_block_sum(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1,):
        super().__init__()
        self.t_min_prev, self.t_min_1, self.t_max_1=0, 0, 1
        self.B_n = X_n #(1 + 0.5) * X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None
        # self.linear = nn.Linear(dim_in, dim_out)

        self.D_i = 0#nn.Parameter(torch.zeros(dim_out,))
        # self.weight = nn.Parameter(torch.ones((dim_in, dim_out)) , requires_grad=False)
        self.register_buffer("weight", torch.ones((dim_in, dim_out)))

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
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer.
        """
        self.threshold = self.t_max_2 - self.t_min_1 - self.D_i
        z =  t_j @ self.weight
        t_i = torch.where(z < self.B_n, z, self.B_n)
        return t_i



'''
Mean
'''
class TTFS_block_mean(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1,):
        super().__init__()

        self.t_min_prev, self.t_min_1, self.t_max_1=0, 0, 1
        self.B_n = X_n #(1 + 0.5) * X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None
        self.linear = nn.Linear(dim_in, dim_out)
        # with torch.no_grad():
        #     self.linear.weight.fill_(1.0/dim_in)  # 1/N
        #     self.linear.bias.fill_(0.0)
        self.D_i = 0#nn.Parameter(torch.zeros(dim_out,))
        # self.weight = nn.Parameter(torch.ones((dim_in, dim_out)) / dim_in, requires_grad=False)
        self.register_buffer("weight", torch.ones((dim_in, dim_out))/ dim_in)

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
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer.
        """
        self.threshold = self.t_max_2 - self.t_min_1 - self.D_i
        z =  (t_j) @ self.weight  #- self.threshold - self.t_min_2# - self.t_max_2 + self.t_max_2
        t_i = torch.where(z < self.B_n, z, self.B_n)
        return t_i


'''
Subtraction
'''
class TTFS_block_sub(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1,):
        super().__init__()
        self.t_min_prev, self.t_min_1, self.t_max_1=0, 0, 1 # TODO!  two time bins
        self.B_n = X_n #(1 + 0.5) * X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None
        self.D_i = 0#nn.Parameter(torch.zeros(dim_out,))
        # self.linear = nn.Linear(dim_in, dim_in)
        # self.weight = nn.Parameter(torch.eye(dim_in) , requires_grad=False)
        self.register_buffer("weight", torch.eye(dim_in))

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

    def forward(self, delta_t):
        """
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer.
        """
        # self.threshold = self.t_max_2 - self.t_min_1 - self.D_i
        z =  (delta_t) @ self.weight  #- self.threshold - self.t_min_2# - self.t_max_2 + self.t_max_2

        t_i = torch.where(z < self.B_n, z, self.B_n)  # !!!!
        return t_i


'''
Square
'''
class TTFS_block_sqr(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1,):
        super().__init__()
        self.t_min_prev, self.t_min_1, self.t_max_1=0, 0, 1
        self.B_n = X_n #(1 + 0.5) * X_n
        self.t_min_2 = self.t_max_1
        self.t_max_2 = self.t_max_1 + self.B_n
        self.threshold = None
        self.D_i = 0#nn.Parameter(torch.zeros(dim_out,))
        # self.linear = nn.Linear(dim_in, dim_in)
        # self.weight = nn.Parameter(torch.eye(dim_in) , requires_grad=False)
        self.register_buffer("weight", torch.eye(dim_in))

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

    def forward(self, t_sub):
        """
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer.
        """
        # self.threshold = self.t_max_2 - self.t_min_1 - self.D_i
        x = (t_sub) ** 2
        z =  -x @ self.weight #- self.threshold - self.t_min_2# - self.t_max_2 + self.t_max_2
        t_i = torch.where(z < 0, z, 0)  # !!!!
        return t_i


'''
variance
'''
class TTFS_block_vmean(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1,):
        super().__init__()
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.B_n = X_n #(1 + 0.5) * X_n
        self.threshold = None
        self.D_i = 0#nn.Parameter(torch.zeros(dim_out,))
        # self.weight = nn.Parameter(torch.ones((dim_in, dim_out)) / dim_in, requires_grad=False)
        self.register_buffer("weight", torch.ones((dim_in, dim_out)) / dim_in)

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min = t_min
        self.t_max =t_min + self.B_n
        return self.t_min, self.t_max

    def forward(self, t_j):
        """
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer.
        """
        # self.threshold = self.t_max - self.t_min - self.D_i
        #print(self.D_i.min(), self.D_i.max())
        # self.D_i = 0
        z =  (t_j) @ self.weight#- self.threshold - self.t_min_2# - self.t_max_2 + self.t_max_2
        t_i = torch.where(z < self.B_n, z, self.B_n)
        return t_i


'''
root
'''
class TTFS_block(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1,):
        super().__init__()
        self.skip_init = True
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.B_n = (1 + 0.5) * X_n
        self.threshold = None
        self.D_i = nn.Parameter(torch.zeros(dim_out,))
        self.linear = skip_init(nn.Linear, dim_in, dim_out)
        self.linear._no_reinit = True
        self.linear.is_fixed = True

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min = t_min
        self.t_max =t_min + self.B_n
        return self.t_min, self.t_max

    def forward(self, t_j):
        """
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer.
        """
        t_j = t_j.to(self.linear.weight.dtype)
        self.threshold = self.t_max - self.t_min - self.D_i
        z =  self.linear(t_j)  - self.D_i #+ self.threshold
        t_i = torch.where(z < 0, z, 0)
        return t_i

class TTFS_out(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1,):
        super().__init__()
        self.skip_init = True
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.B_n = (1 + 0.5) * X_n
        self.threshold = None
        self.D_i = nn.Parameter(torch.zeros(dim_out,))
        self.alpha = torch.ones((dim_out,))
        self.linear = skip_init(nn.Linear, dim_in, dim_out)#nn.Linear(dim_in, dim_out)
        self.linear._no_reinit = True
        self.linear.is_fixed = True

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min = t_min
        self.t_max =t_min + self.B_n
        return self.t_min, self.t_max

    def forward(self, t_j):
        """
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer.
        """
        c = self.t_max - self.t_min - self.D_i
        # Read out the value of membrane potential at time t_min.
        z =  self.linear(-t_j) #+ self.threshold + self.t_min
        self.alpha = self.D_i/(self.t_min - self.t_min_prev)
        output = self.alpha * (self.t_min - self.t_min_prev) + z
        #output = torch.where(output < self.t_max, output, self.t_max)
        return output

class TTFS_SNN_root(nn.Module):
    def __init__(self, X_n):
        super().__init__()
        self.skip_init = True
        self.layers = nn.ModuleList(
            [TTFS_block(1, 512, X_n)]
        )
        # self.min_ti = []
        self.out = TTFS_out(512, 1, 1)

    def forward(self, x):
        self.min_ti = []
        for layer in self.layers:
            x = layer(x)
            # self.min_ti.append(x.min(dim=-1).values)
        y = self.out(x)
        return  y



'''
normalization
'''
class TTFS_SNN_norm(nn.Module):
    def __init__(self, dim_in, dim_out=1, X_n=500):
        super().__init__()
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.B_n = (1 + 0.5) * X_n
        self.threshold = None
        self.ttfs_root0 = TTFS_SNN_root(X_n)

        self.ttfs_root0.load_state_dict(torch.load('/home/ubuntu/zoe/TTFS_GPT2/weight_base/model_weights_norm_gpt0.pth', map_location='cuda:0'))    #TODO
        self.ttfs_root1 = TTFS_SNN_root(X_n)
        self.ttfs_root1.load_state_dict(torch.load('/home/ubuntu/zoe/TTFS_GPT2/weight_base/model_weights_norm_gpt1.pth', map_location='cuda:0'))    #TODO
        self.ttfs_root2 = TTFS_SNN_root(X_n)
        self.ttfs_root2.load_state_dict(torch.load('/home/ubuntu/zoe/TTFS_GPT2/weight_base/model_weights_norm_gpt2.pth', map_location='cuda:0'))
        self.ttfs_root3 = TTFS_SNN_root(X_n)
        self.ttfs_root3.load_state_dict(torch.load('/home/ubuntu/zoe/TTFS_GPT2/weight_base/model_weights_norm_gpt7.pth', map_location='cuda:0'))


        for p in self.ttfs_root1.parameters():
            p.requires_grad = False
        for p in self.ttfs_root2.parameters():
            p.requires_grad = False

        self.layers = nn.ModuleList(
            [TTFS_block_mean(dim_in, dim_out, X_n)]   +   #mean spike time
            [TTFS_block_sub(dim_in, dim_out, X_n)]    +   #sub      spike time
            [TTFS_block_sqr(dim_in, dim_out, X_n)]    +     #sqr     spike time
            [TTFS_block_vmean(dim_in, dim_out, X_n)]  +      # variance  spike time
            [self.ttfs_root0] + [self.ttfs_root1] + [self.ttfs_root2]  + [self.ttfs_root3]                    #root 1/root(x)   membrane potential
        )
        # print() # Fine

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min = t_min
        self.t_max =t_min + self.B_n
        return self.t_min, self.t_max

    def forward(self, t_in):
        t_mu = self.layers[0](t_in)

        delta_t = t_in - t_mu
        t_sub = self.layers[1](delta_t)
        x = t_sub
        t_sqr = self.layers[2](t_sub)

        t_var = self.layers[3](t_sqr)

        self.delta_t = (0 - t_var) + 1e-10

        ## TODO ######### estimation
        mask = self.delta_t < 0.005
        mask0 = (self.delta_t >= 0.005) & (self.delta_t < 0.1)
        mask1 = (self.delta_t >= 0.1) & (self.delta_t < 50)
        mask2 = (self.delta_t >= 50) & (self.delta_t < 3000)

        t_root_em = torch.where(
            mask, self.ttfs_root0(self.delta_t), torch.where(
                mask0, self.ttfs_root1(self.delta_t), torch.where(
                    mask1, self.ttfs_root2(self.delta_t), torch.where(
                        mask2, self.ttfs_root3(self.delta_t)/1000, 0.
                    )
                )
            )
        )

        ## TODO ######### real
        # t_root = 1/torch.sqrt(self.delta_t)

        x_norm = t_root_em * x   # TODO
        t_i = torch.where(x_norm <  self.B_n, x_norm, self.B_n)

        return t_i


# '''
# 1/sqrt(x) # hidden 256
# '''
# model_sx = TTFS_SNN_root(X_n=500).to('cuda:1')
# t_min, t_max = 1, 2   #8501.5, 10001.5  #5001.5, 6001.5  # for the input layer
# for layer in model_sx.layers:
#     t_min, t_max = layer.set_params(t_min, t_max)
# t_min, t_max = model_sx.out.set_params(t_min, t_max)

# # generate data
# x_train = torch.linspace(50, 3000, 50000).reshape(-1, 1).to('cuda:1')
# y_train = 1000 / torch.sqrt(x_train).to('cuda:1')
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model_sx.parameters(), lr=0.001)
#
# epochs = 100000
# for epoch in range(epochs):
#     indices = torch.randperm(50000)[:500]
#
#     optimizer.zero_grad()
#     outputs = model_sx(x_train[indices])
#     loss = criterion(outputs, y_train[indices])
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 500 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
#
# torch.save(model_sx.state_dict(), '/home/zoe/TTFS_GPT2/weight_base/model_weights_norm_gpt7.pth')
# model_sx.load_state_dict(torch.load('/home/zoe/TTFS_GPT2/weight_base/model_weights_norm_gpt0.pth'))

# test
# x_test = torch.linspace(0.0005, 0.0011, 100).reshape(-1, 1)
# y_pred = model_sx(x_test.to('cuda:1')).detach().cpu().numpy()
# print(y_pred)
# # 绘制对比图
# plt.figure(figsize=(10, 5))
# plt.plot(x_test, 1/np.sqrt(x_test), label='True: $1/\sqrt{x}$')
# plt.plot(x_test, y_pred, '--', label='SNN Prediction')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.title('Fitting $1/\sqrt{x}$ with a 2-Layer TTFS Network')
# plt.show()
