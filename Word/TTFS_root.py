
from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TTFS_block(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1,):
        super().__init__()
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.B_n = (1 + 0.5) * X_n
        self.threshold = None
        self.D_i = nn.Parameter(torch.zeros(dim_out,))
        self.linear = nn.Linear(dim_in, dim_out)
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
        self.threshold = self.t_max - self.t_min - self.D_i
        #print(self.D_i.min(), self.D_i.max())
        # z =  self.linear(t_j - self.t_min) + self.threshold + self.t_min
        z =  self.linear(t_j) + self.threshold + self.t_min
        t_i = torch.where(z < self.t_max, z, self.t_max)
        return t_i

class TTFS_out(nn.Module):
    def __init__(self, dim_in, dim_out, X_n=1,):
        super().__init__()
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.B_n = (1 + 0.5) * X_n
        self.threshold = None
        self.D_i = nn.Parameter(torch.zeros(dim_out,))
        self.alpha = torch.ones((dim_out,))
        self.linear = nn.Linear(dim_in, dim_out)
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
        z =  self.linear(self.t_min - t_j) #+ self.threshold + self.t_min
        self.alpha = self.D_i/(self.t_min-self.t_min_prev)
        output = self.alpha * (self.t_min - self.t_min_prev) + z
        # output = torch.where(output < self.t_max, output, self.t_max)
        return output

class TTFS_SNN_root(nn.Module):
    def __init__(self, X_n):
        super().__init__()
        self.layers = nn.ModuleList(
            [TTFS_block(1, 512, X_n)]
        )
        self.min_ti = []
        #self.out = TTFS_block(128, 1, 1)
        self.out = TTFS_out(512, 1, 1)


    def forward(self, x):
        self.min_ti = []
        for layer in self.layers:
            x = layer(x)
            self.min_ti.append(x.min(dim=-1).values)

        y = self.out(x)
        # y =  self.layers[-1].t_max - x
        return  y

device = "cuda:0"

# '''
# 1/sqrt(x) # hidden 256
# '''
model_sx = TTFS_SNN_root(X_n=1000).to(device)
# t_min, t_max = 0, 1  # for the input layer
t_min, t_max = 6000, 7000   #8501.5, 10001.5  #5001.5, 6001.5  # for the input layer
for layer in model_sx.layers:
    t_min, t_max = layer.set_params(t_min, t_max)
t_min, t_max = model_sx.out.set_params(t_min, t_max)
#
# # generate data
x_train = torch.linspace(50, 3000, 50000).reshape(-1, 1).to(device)
y_train = 1 / torch.sqrt(x_train).to(device)
# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model_sx.parameters(), lr=0.005)

# epochs = 20500
# for epoch in range(epochs):
#     indices = torch.randperm(50000)[:5000]
#
#     optimizer.zero_grad()
#     outputs = model_sx(x_train[indices])
#     loss = criterion(outputs, y_train[indices]*100)
#     # loss = ((y_train[indices] - outputs) / y_train[indices]).pow(4).mean()
#
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 50 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# torch.save(model_sx.state_dict(), '/home/zoe/TTFS-BERT/model/BasicBert/LN_base_cache/model_weights_norm_gpt7.pth')
model_sx.load_state_dict(torch.load('/home/zoe/TTFS-BERT/model/BasicBert/LN_base_cache/model_weights_norm_gpt7.pth'))

# # test
x_test = torch.linspace(50, 3000, 50000).reshape(-1, 1)
y_pred = model_sx(x_test.to(device)).detach().cpu().numpy()
# print(y_pred)
# 绘制对比图
print(y_pred[0]/100)
plt.figure(figsize=(10, 5))
plt.plot(x_test, 1/np.sqrt(x_test), label='True: $1/\sqrt{x}$')
plt.plot(x_test, y_pred/100, '--', label='SNN Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Fitting $1/\sqrt{x}$ with a 2-Layer TTFS Network')
plt.savefig("1")

"""
e^{x} hidden 256
"""
# model_ex = TTFS_SNN(X_n=1000).to(device)
# t_min, t_max = 0, 1  # for the input layer
# for layer in model_ex.layers:
#     t_min, t_max = layer.set_params(t_min, t_max)
# t_min, t_max = model_ex.out.set_params(t_min, t_max)

# x_train = torch.linspace(-10, 10, 10000).reshape(-1, 1).to(device)
# y_train = torch.exp(x_train).to(device)
#
# # model = TwoLayerNet(hidden_size=128)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model_ex.parameters(), lr=0.001)

# epochs = 300000
# for epoch in range(epochs):
#     indices = torch.randperm(10000)[:1000]
#
#     optimizer.zero_grad()
#     outputs = model_ex(x_train[indices])
#     loss = criterion(outputs, y_train[indices])
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 5000 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# torch.save(model_ex.state_dict(), '/home/zoe/nanoGPT/base_weight/model_weights_ex.pth')
# model_ex.load_state_dict(torch.load('/home/zoe/nanoGPT/base_weight/model_weights_ex.pth'))
# x_test = torch.linspace(-10, 10, 100).reshape(-1, 1)
# y_pred = model_ex(x_test.to(device)).detach().cpu().numpy()
# #

#
# plt.figure(figsize=(10, 5))
# plt.plot(x_test, torch.exp(x_test), label='True: $e^{x}$')
# plt.plot(x_test, y_pred, '--', label='SNN Prediction')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.title('Fitting $e^{x}$ with a 2-Layer TTFS Network')
# plt.show()