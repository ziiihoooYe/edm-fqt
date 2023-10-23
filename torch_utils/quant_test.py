import os
import sys
sys.path.append(os.getcwd())
from training.networks import Linear, Conv2d, AttentionOp, UNetBlock, SongUNet
from quantize import QuantizationConfig, conv2d_q, linear_q
import quantize
import matplotlib.pyplot as plt
import numpy as np
import torch
sys.path.append(os.getcwd())
sys.path.append('/home/yezihao-fwxz/edm')
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"


# flatten
def flatten(arr):
    return np.concatenate([grad.flatten() for grad in arr])


# min=0.025， max=0.975
def percent_range(dataset, min=0.03, max=0.97):
    range_max = np.percentile(dataset, max * 100)
    range_min = -np.percentile(-dataset, (1 - min) * 100)

    # 剔除前20%和后80%的数据
    new_data = []
    for value in dataset:
        if value < range_max and value > range_min:
            new_data.append(value)
    return new_data


# ------------------------------------------------------------------------------------
### global variable preparation

device = torch.device('cuda')  # decive: cuda
x = torch.rand([1, 16, 128, 128]).to(device)  # (B, C, H, W)
# x = torch.randn([1, 256, 1024]) # (B, C, N)
x.requires_grad = True
layer_num = 2
training_epoch = 5

# ------------------------------------------------------------------------------------
### model preparation

# #linear Quantization
# w = torch.randn(3, 4)
# output = QF.linear(input=x, weight=w)
# net = Linear(16, 16).to('cuda')

# conv quantization
# net = torch.nn.Sequential()
# for i in range(layer_num):
#     net.append(Conv2d(16, 16, 3))
# net.to(device)

# U-Net
# net = UNetBlock(16, 16, 128*4).to(device)

# song U-Net
net = SongUNet(128, 16, 16).to(device)

# ------------------------------------------------------------------------------------
### training

for epoch in range(training_epoch):
# forward
# fully quantization traning
    x_temp = x
    for i in range(layer_num):
        # x_temp = net(x)
        # x_temp = convs[i](x_temp)
        # x_temp = torch.nn.functional.silu(x_temp)
        # emb = torch.randn(1, 512).to(device)
        noise_labels = torch.randn(1,).to(device)
        class_labels = torch.randn(1, 0).to(device)
        x_temp = net(x_temp, noise_labels, class_labels)
    output_q = x_temp

    # set target
    target = torch.rand(output_q.size()).to(device)

    # loss
    loss = torch.nn.MSELoss()(output_q, target)

    # collect grads
    loss.backward()
    
    for name, param in net.named_parameters():
        if param.grad is None:
            print(name)
            
    print()
# quant_grad_list = flatten(QuantizationConfig.grad)
# quant_grad_var = QuantizationConfig.grad_var
# quant_grad_list = percent_range(quant_grad_list)
# plt.hist(quant_grad_list, color='red', label='quantized grads distribution', bins=100, alpha=0.5)


# #clear the grads list
# QuantizationConfig.clear_grad()

# #quantization aware training
# QuantizationConfig.fake_quant = True
# x_temp = x

# #forward
# for i in range(layer_num):
#     x_temp = QF.quantize(x_temp, num_bits=8)  #activation quantization
#     x_temp = convs[i](x_temp)
#     x_temp = torch.nn.functional.relu(x_temp)
# output = x_temp

# loss = torch.nn.MSELoss()(output, target)
# loss.backward()
# QuantizationConfig.grad = flatten(QuantizationConfig.grad)
# grad_var = QuantizationConfig.grad_var
# grad_list = flatten(QuantizationConfig.grad)
# # grad_list = percent_range(grad_list)
# plt.hist(grad_list, color='blue', label='original grads distribution', bins=100, alpha=0.5)

# plt.savefig(r'fig/conv_16layer.png')
# print('quant_grad_var: ' + str(quant_grad_var))
# print('grad_var: ' + str(grad_var))
