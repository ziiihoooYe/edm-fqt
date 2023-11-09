from collections import namedtuple
import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn.functional import conv2d
from torch_utils import persistence
from torch_utils.distributed import print0


class QuantizationConfig:
    # quantization type
    quantize_activation = True
    quantize_weights = True
    quantize_gradient = True
    switch_back = True

    # quantization bit number
    activation_num_bit = 32 # 32
    weight_num_bit = 32 # 32
    bias_num_bit = 16
    backward_num_bit = 8
    bweight_num_bit = 32

    # quantization choice
    per_channel = True
    per_sample = True

    # class grads list
    grad = None  # None if do not collect grads / []
    grad_n = 0
    grad_mean = 0
    grad_M2 = 0
    grad_var = None
    grad_drop_r = 0.95  # probability that welford update drop the grads

    # batch init
    batch_init = 20  # 20

    # stochastical sample random grads from grads_list and update the grads stat identifiers

    @staticmethod
    def welford_update(grads_list):
        for grads in grads_list:
            grads = grads.flatten()  # flatten each multi-dimension array
            for grad in grads:
                prob = random.random()
                if (prob > QuantizationConfig.grad_drop_r):
                    QuantizationConfig.grad_n += 1
                    delta = grad - QuantizationConfig.grad_mean
                    QuantizationConfig.grad_mean += delta / QuantizationConfig.grad_n
                    delta2 = grad - QuantizationConfig.grad_mean
                    QuantizationConfig.grad_M2 += delta * delta2

        QuantizationConfig.grad_var = QuantizationConfig.grad_M2 / QuantizationConfig.grad_n

    @staticmethod
    def add_grad(grads):
        if QuantizationConfig.grad is not None:
            # QuantizationConfig.grad.append(grads.detach().cpu().numpy())  #add to grad list
            QuantizationConfig.welford_update(
                grads.detach().cpu().numpy())  # update grad stat identifiver


class Round(Function):
    @staticmethod
    def forward(ctx, _input):
        output = torch.round(_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


#----------------------------------------------------------------------------
# unified routine for initializing weights and biases. (by edm)
def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


#----------------------------------------------------------------------------
# conv2d grad quantizer (per-sample option) - (B, C, H, W)
def grad_quantize_conv(grad, num_bit_grad, per_sample=False):
    if per_sample:
        # calculate the maximum and minimum values of each sample on the B dimension
        max_vals, _ = grad.reshape(grad.size(0), -1).max(1)
        min_vals, _ = grad.reshape(grad.size(0), -1).min(1)
        
        # calculate div_grad and zero_point
        div_grad = (2 ** num_bit_grad) / (max_vals - min_vals + 1e-9)
        zero_point = min_vals
        
        # extend the shape to match grad_output
        div_grad = div_grad.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    else:
        max_vals = grad.max()
        min_vals = grad.min()
        
        # calculate div_grad and zero_point
        div_grad = (2 ** num_bit_grad) / (max_vals - min_vals + 1e-9)
        zero_point = min_vals
        
    # scale gradient and round
    grad = ((Round.apply((grad - zero_point) * div_grad)) / div_grad) + zero_point
    
    return grad


#----------------------------------------------------------------------------
# linear grad quantizer (per-sample option) - (B, C)
def grad_quantize_linear(grad, num_bit_grad, per_sample=False):
    if per_sample:
        # calculate the maximum and minimum values of each sample in dimension B
        max_vals, _ = grad.reshape(grad.size(0), -1).max(1)
        min_vals, _ = grad.reshape(grad.size(0), -1).min(1)
        
        # calculate div_grad and zero_point
        div_grad = (2 ** num_bit_grad) / (max_vals - min_vals + 1e-9)
        zero_point = min_vals
        
        # expand the shape to match grad_output
        div_grad = div_grad.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        max_vals = grad.max()
        min_vals = grad.min()
        
        # calculate div_grad and zero_point
        div_grad = (2 ** num_bit_grad) / (max_vals - min_vals + 1e-9)
        zero_point = min_vals
        
    # scale the gradient and round it
    grad = ((Round.apply((grad - zero_point) * div_grad)) / div_grad) + zero_point
    
    return grad


#----------------------------------------------------------------------------
# forward activation quantization: 
# a -> activation, 
# s -> scale, 
# b -> bias, 
# num_bit -> forward activation number of bit
def activation_quantize(a, s, b, num_bit):
    if num_bit == 32:
        # full-precision forward activation
        a_q = a
    else:
        # quantized forward activation
        Qn = -(2 ** (num_bit - 1))
        Qp = (2 ** (num_bit - 1)) - 1
        a_q = Round.apply(torch.div((a - b), (s + 1e-9)).clamp(Qn, Qp))
        a_q = a_q * s + b
    return a_q


#----------------------------------------------------------------------------
# backward activation quantization: 
# a -> activation, 
# s -> scale, 
# b -> bias, 
# g -> step size, 
# num_bit -> forward activation number of bit
def activation_quantizer_backward(grad_act, a, s, b, g, num_bit):
    if num_bit == 32:
        # full-precision backward activation/activation_quantizer_bias/activation_quantizer_scale gradient
        grad_act = grad_act
        grad_b = torch.Tensor([0]).cuda()
        grad_s = torch.Tensor([0]).cuda()
    else:
        # quantized backward activation/activation_quantizer_bias/activation_quantizer_scale gradient
        Qn = -(2 ** (num_bit - 1))
        Qp = (2 ** (num_bit - 1)) - 1
        q_a = (a - b) / (s + 1e-9)
        smaller = (q_a < Qn).float()  # 1.0/0.0
        bigger = (q_a > Qp).float()  # 1.0/0.0
        between = 1.0 - smaller - bigger  # 1.0/0.0

        grad_s = ((smaller * Qn + bigger * Qp +
                between * Round.apply(q_a) -
                between * q_a) * grad_act * g).sum().unsqueeze(dim=0)
        grad_b = ((smaller + bigger) * grad_act * g).sum().unsqueeze(dim=0)

        grad_act = between * grad_act

    return grad_act, grad_s, grad_b


#----------------------------------------------------------------------------
# forward weight quantization: 
# w -> weight, 
# s -> scale, 
# num_bit -> forward weight number of bit (per-channel option)
def weight_quantize(w, s, num_bit, per_channel):
    if num_bit == 32:
        # full-precision forward weight
        w_q = w
    else:
        # quantized forward weight
        Qn = -(2 ** (num_bit - 1))
        Qp = (2 ** (num_bit - 1)) - 1

        # per-channel option
        if per_channel:
            sizes = w.size()
            w = w.contiguous().view(w.size()[0], -1)
            w = torch.transpose(w, 0, 1)
            s = torch.broadcast_to(s, w.size())
            w_q = Round.apply(torch.div(w, (s + 1e-9)).clamp(Qn, Qp))
            w_q = w_q * s
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(w, (s + 1e-9)).clamp(Qn, Qp))
            w_q = w_q * s
    return w_q

#----------------------------------------------------------------------------
# backward weight quantization: 
# w-> weight, 
# s -> scale, 
# g -> step, 
# num_bit -> forward weight number of bit (per-channel option)
def weight_quantizer_backward(grad_weight, w, s, g, num_bit, per_channel):
    s_size = s.size()

    # collect gradients if needs
    if QuantizationConfig.grad is not None:
        QuantizationConfig.add_grad(grad_weight)

    if num_bit == 32:
        # full-precision backward weight/weight_quantizer_scale gradient
        grad_weight = grad_weight
        grad_s = torch.Tensor([0]).expand(s.size()).cuda()
    else:
        # quantized backward weight/weight_quantizer_scale gradient
        Qn = -(2 ** (num_bit - 1))
        Qp = (2 ** (num_bit - 1)) - 1
        
        # per-channel option
        if per_channel:
            sizes = w.size()
            w = w.contiguous().view(w.size()[0], -1)
            w = torch.transpose(w, 0, 1)
            s = torch.broadcast_to(s, w.size())
            q_w = w / (s + 1e-9)
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = w / (s + 1e-9)

        smaller = (q_w < Qn).float()  # 1.0/0.0
        bigger = (q_w > Qp).float()  # 1.0/0.0
        between = 1.0 - smaller - bigger  # 1.0/0.0
        if per_channel:
            grad_s = ((smaller * Qn +
                    bigger * Qp +
                    between * Round.apply(q_w) - between * q_w)*grad_weight * g)
            grad_s = grad_s.contiguous().view(grad_s.size()[0], -1).sum(dim=1)
            grad_s = torch.broadcast_to(grad_s, s_size)
        else:
            grad_s = ((smaller * Qn +
                    bigger * Qp +
                    between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
            grad_s = torch.broadcast_to(grad_s, s_size)

        grad_weight = between * grad_weight

    return grad_weight, grad_s


#----------------------------------------------------------------------------
# activation quantizer initialization state update function: 
# s -> scale, 
# b -> bias, 
# g -> step, 
# init_stat -> initialization state (< batch_init -> initialization, = batch_init -> initialization done),
# num_bit -> forward activation number of bit,
# batch_init -> batch initialization number max
def activation_quantizer_init_update(activation, s, b, g, num_bit, init_stat, batch_init = QuantizationConfig.batch_init):
    # initialization state == 0 -> initialize the scale and bias
    if init_stat.item() == 0:
        Qp = (2 ** (num_bit - 1)) - 1
        Qn = -(2 ** (num_bit - 1))
        g = 1.0/math.sqrt(activation.numel() * Qp)
        
        # Compute the maximum and minimum of the activation.
        max_a = activation.detach().max()
        min_a = activation.detach().min()

        # initialize the scale and bias
        s.data = ((max_a - min_a + 1e-9)/(Qp - Qn)).unsqueeze(0)
        b.data = min_a - s.data * Qn

        # update the initialization state
        init_stat += 1
        init_stat = init_stat.to(s.device)

    # initialization state < batch_init -> update the scale and bias
    elif init_stat.item() < batch_init:
        Qp = (2 ** (num_bit - 1)) - 1
        Qn = -(2 ** (num_bit - 1))
        
        # Compute the maximum and minimum of the activation.
        max_a = activation.detach().max()
        min_a = activation.detach().min()

        # update the scale and bias
        s.data = s.data*0.9 + 0.1 * (max_a - min_a + 1e-9)/(Qp - Qn)
        b.data = s.data*0.9 + 0.1 * (min_a - s.data * Qn)
        
        # update the initialization state
        init_stat += 1

    # initialization state == batch_init -> initialization done
    elif init_stat.item() == batch_init:

        # update the initialization state
        init_stat += 1        
        
    return s, b, g, init_stat


#----------------------------------------------------------------------------
# weight quantizer initialization state update function: 
# s -> scale, 
# g -> step, 
# init_stat -> initialization state (< batch_init -> initialization, = batch_init -> initialization done),
# num_bit -> forward weight number of bit,
# batch_init -> batch initialization number max,
# per_channel -> per-channel option
def weight_quantizer_init_update(weight, s, g, num_bit, init_stat, per_channel, batch_init = QuantizationConfig.batch_init):
    # initialization state == 0 -> initialize the scale
    if init_stat == 0:
        Qp = (2 ** (num_bit - 1)) - 1
        div = 2 ** num_bit - 1
        g = 1.0/math.sqrt(weight.numel() * Qp)

        # initialize the scale (per-channel option)
        if per_channel:
            weight_tmp = weight.detach().contiguous().view(
                weight.size()[0], -1)
            mean = torch.mean(weight_tmp, dim=1)
            std = torch.std(weight_tmp, dim=1)
            s.data, _ = torch.max(torch.stack(
                [torch.abs(mean-3*std), torch.abs(mean + 3*std)]), dim=0)
            s.data = s.data / div
        else:
            mean = torch.mean(weight.detach())
            std = torch.std(weight.detach())
            s.data = (
                max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / div).unsqueeze(0)
            
        # update the initialization state
        init_stat += 1
        init_stat = init_stat.to(s.device)

    # initialization state < batch_init -> update the scale
    elif init_stat < batch_init:
        Qp = (2 ** (num_bit - 1)) - 1
        div = 2 ** num_bit - 1

        # update the scale (per-channel option)
        if per_channel:
            weight_tmp = weight.detach().contiguous().view(
                weight.size()[0], -1)
            mean = torch.mean(weight_tmp, dim=1)
            std = torch.std(weight_tmp, dim=1)
            s.data, _ = torch.max(torch.stack(
                [torch.abs(mean-3*std), torch.abs(mean + 3*std)]), dim=0)
            s.data = s.data*0.9 + 0.1*s.data / div
        else:
            mean = torch.mean(weight.detach())
            std = torch.std(weight.detach())
            s.data = s.data*0.9 + 0.1 * \
                max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / div
                
        # update the initialization state
        init_stat += 1

    # initialization state == batch_init -> initialization done
    elif init_stat == batch_init:
        
        # update the initialization state
        init_stat += 1
        
    return s, g, init_stat


#----------------------------------------------------------------------------
# deprecated activation/weight quantizer class
# class QuantizeA(Function):
#     @staticmethod
#     def forward(ctx, a, s, b, g, num_bit):
#         ctx.save_for_backward(a, s, b)
#         ctx.other = g, num_bit

#         if num_bit == 32:
#             a_q = a
#         else:
#             # quantize input
#             Qn = -(2 ** (num_bit - 1))
#             Qp = (2 ** (num_bit - 1)) - 1
#             a_q = Round.apply(torch.div((a - b), (s + 1e-9)).clamp(Qn, Qp))
#             a_q = a_q * s + b

#         return a_q

#     @staticmethod
#     def backward(ctx, grad_act):
#         a, s, b = ctx.saved_tensors
#         g, num_bit = ctx.other

#         if num_bit == 32:
#             grad_act = grad_act
#             grad_b = torch.Tensor([0]).cuda()
#             grad_s = torch.Tensor([0]).cuda()
#         else:
#             Qn = -(2 ** (num_bit - 1))
#             Qp = (2 ** (num_bit - 1)) - 1
#             q_a = (a - b) / (s + 1e-9)
#             smaller = (q_a < Qn).float()  # 1.0/0.0
#             bigger = (q_a > Qp).float()  # 1.0/0.0
#             between = 1.0 - smaller - bigger  # 1.0/0.0

#             # quantization scale and bias gradients
#             grad_s = ((smaller * Qn + bigger * Qp +
#                     between * Round.apply(q_a) -
#                     between * q_a) * grad_act * g).sum().unsqueeze(dim=0)
#             grad_b = ((smaller + bigger) * grad_act * g).sum().unsqueeze(dim=0)

#             grad_act = between * grad_act

#         return grad_act, grad_s, grad_b, None, None


# class QuantizeW(Function):
#     @staticmethod
#     def forward(ctx, w, s, g, num_bit, per_channel):
#         ctx.save_for_backward(w, s)
#         ctx.other = g, num_bit, per_channel

#         if num_bit == 32:
#             w_q = w
#         else:
#             Qn = -(2 ** (num_bit - 1))
#             Qp = (2 ** (num_bit - 1)) - 1

#             # quantize weight
#             if per_channel:
#                 sizes = w.size()
#                 w = w.contiguous().view(w.size()[0], -1)
#                 w = torch.transpose(w, 0, 1)
#                 s = torch.broadcast_to(s, w.size())
#                 w_q = Round.apply(torch.div(w, (s + 1e-9)).clamp(Qn, Qp))
#                 w_q = w_q * s
#                 w_q = torch.transpose(w_q, 0, 1)
#                 w_q = w_q.contiguous().view(sizes)
#             else:
#                 w_q = Round.apply(torch.div(w, (s + 1e-9)).clamp(Qn, Qp))
#                 w_q = w_q * s
#         return w_q

#     @staticmethod
#     def backward(ctx, grad_weight):                
#         w, s = ctx.saved_tensors
#         g, num_bit, per_channel = ctx.other

#         s_size = s.size()

#         # collect gradients if needs
#         if QuantizationConfig.grad is not None:
#             QuantizationConfig.add_grad(grad_weight)

#         if num_bit == 32:
#             grad_weight = grad_weight
#             grad_s = torch.Tensor([0]).expand(s.size()).cuda()
#         else:
#             Qn = -(2 ** (num_bit - 1))
#             Qp = (2 ** (num_bit - 1)) - 1
#             if per_channel:
#                 sizes = w.size()
#                 w = w.contiguous().view(w.size()[0], -1)
#                 w = torch.transpose(w, 0, 1)
#                 s = torch.broadcast_to(s, w.size())
#                 q_w = w / (s + 1e-9)
#                 q_w = torch.transpose(q_w, 0, 1)
#                 q_w = q_w.contiguous().view(sizes)
#             else:
#                 q_w = w / (s + 1e-9)

#             smaller = (q_w < Qn).float()  # 1.0/0.0
#             bigger = (q_w > Qp).float()  # 1.0/0.0
#             between = 1.0 - smaller - bigger  # 1.0/0.0
#             if per_channel:
#                 grad_s = ((smaller * Qn +
#                         bigger * Qp +
#                         between * Round.apply(q_w) - between * q_w)*grad_weight * g)
#                 # print('s0: ' + str(grad_s.size()))
#                 grad_s = grad_s.contiguous().view(grad_s.size()[0], -1).sum(dim=1)
#                 # print('s1: ' + str(grad_s.size()))
#                 grad_s = torch.broadcast_to(grad_s, s_size)
#                 # print('s2: ' + str(grad_s.size()))
#             else:
#                 grad_s = ((smaller * Qn +
#                         bigger * Qp +
#                         between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
#                 grad_s = torch.broadcast_to(grad_s, s_size)

#             grad_weight = between * grad_weight

#         return grad_weight, grad_s, None, None, None


# class ActivationQuantizer(nn.Module):
#     def __init__(self, num_bit, num_bit_grad=None, batch_init=QuantizationConfig.batch_init):
#         super(ActivationQuantizer, self).__init__()
#         self.num_bit = num_bit
#         self.num_bit_grad = num_bit_grad
#         self.batch_init = batch_init
#         self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
#         self.b = torch.nn.Parameter(torch.ones(1), requires_grad=True)

#         self.init_stat = torch.Tensor([0])

#     def forward(self, activation):
#         if self.init_stat.item() == 0:
#             Qp = (2 ** (self.num_bit - 1)) - 1
#             Qn = -(2 ** (self.num_bit - 1))
#             self.g = 1.0/math.sqrt(activation.numel() * Qp)
            
#             # Compute the maximum and minimum of the activation.
#             max_a = activation.detach().max()
#             min_a = activation.detach().min()

#             self.s.data = ((max_a - min_a + 1e-9)/(Qp - Qn)).unsqueeze(0)
#             self.b.data = min_a - self.s.data * Qn

#             self.init_stat += 1
#             self.init_stat = self.init_stat.to(self.s.device)

#         elif self.init_stat.item() < self.batch_init:
#             Qp = (2 ** (self.num_bit - 1)) - 1
#             Qn = -(2 ** (self.num_bit - 1))
            
#             # Compute the maximum and minimum of the activation.
#             max_a = activation.detach().max()
#             min_a = activation.detach().min()

#             self.s.data = self.s.data*0.9 + 0.1 * (max_a - min_a + 1e-9)/(Qp - Qn)
#             self.b.data = self.s.data*0.9 + 0.1 * (min_a - self.s.data * Qn)
            
#             self.init_stat += 1

#         elif self.init_stat.item() == self.batch_init:

#             self.init_stat += 1        

#         if QuantizationConfig.quantize_activation:
#             a_q = QuantizeA.apply(activation, self.s, self.b,
#                                   self.g, self.num_bit)
#         else:
#             a_q = QuantizeA.apply(activation, self.s, self.b,
#                                   self.g, 32)
        
#         # print('self.s.grad: ' + str(self.s.grad))
#         # print('self.b.grad: ' + str(self.b.grad))
            
#         return a_q


# class WeightQuantizer(nn.Module):
    def __init__(self, num_bit, num_bit_grad, channel_num=None, per_channel=False, batch_init=QuantizationConfig.batch_init):
        super(WeightQuantizer, self).__init__()
        self.num_bit = num_bit
        self.num_bit_grad = num_bit_grad
        self.batch_init = batch_init
        self.per_channel = per_channel
        self.init_stat = 0
        if self.per_channel:
            self.s = torch.nn.Parameter(torch.ones(channel_num), requires_grad=True)
        else:
            self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, weight):
        if self.init_stat == 0:
            Qp = (2 ** (self.num_bit - 1)) - 1
            div = 2 ** self.num_bit - 1
            self.g = 1.0/math.sqrt(weight.numel() * Qp)

            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(
                    weight.size()[0], -1)
                mean = torch.mean(weight_tmp, dim=1)
                std = torch.std(weight_tmp, dim=1)
                self.s.data, _ = torch.max(torch.stack(
                    [torch.abs(mean-3*std), torch.abs(mean + 3*std)]), dim=0)
                self.s.data = self.s.data / div
                
            else:
                mean = torch.mean(weight.detach())
                std = torch.std(weight.detach())
                self.s.data = (
                    max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / div).unsqueeze(0)
            
            self.init_stat += 1

        elif self.init_stat < self.batch_init:
            Qp = (2 ** (self.num_bit - 1)) - 1
            div = 2 ** self.num_bit - 1

            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(
                    weight.size()[0], -1)
                mean = torch.mean(weight_tmp, dim=1)
                std = torch.std(weight_tmp, dim=1)
                self.s.data, _ = torch.max(torch.stack(
                    [torch.abs(mean-3*std), torch.abs(mean + 3*std)]), dim=0)
                self.s.data = self.s.data*0.9 + 0.1*self.s.data / div
            else:
                mean = torch.mean(weight.detach())
                std = torch.std(weight.detach())
                self.s.data = self.s.data*0.9 + 0.1 * \
                    max([torch.abs(mean-3*std), torch.abs(mean + 3*std)]) / div
                        
            self.init_stat += 1

        elif self.init_stat == self.batch_init:
            self.init_stat += 1
        
        if QuantizationConfig.quantize_weights:
            w_q = QuantizeW.apply(weight, self.s, self.g,
                                  self.num_bit, self.per_channel)
        else: 
            w_q = QuantizeW.apply(weight, self.s, self.g,
                                  32, None)

        return w_q


#----------------------------------------------------------------------------
# custom fqt linear function 
# forward options: lsq+, per_channel weight forward
# backward options: gradient quantization, per_sample gradient backward, switch back
class LinearQ(Function):
    
    @staticmethod
    def forward(ctx, _input, weight, bias,
                a_s, a_b, a_g, a_num_bit, 
                w_s, w_g, w_num_bit, per_channel,
                backward_num_bit, per_sample):
        
        # quantize input & weight - lsq+ option & per_channel weight option
        _input_q = activation_quantize(_input, a_s, a_b, a_num_bit)
        weight_q = weight_quantize(weight, w_s, w_num_bit, per_channel)
        
        # save param for backward prop
        ctx.save_for_backward(_input_q, weight_q, bias, a_s, a_b, w_s)
        ctx.other = _input, weight 
        ctx.a_param = a_g, a_num_bit 
        ctx.w_param = w_g, w_num_bit, per_channel
        ctx.back_param = backward_num_bit, per_sample
        
        # linear manipulation
        output = _input_q @ weight_q.to(_input_q.dtype).t()
        
        # bias manipulation
        if bias is not None:
            output = output.add_(bias.to(_input.dtype))
            
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # obtain saved tensors/params
        _input_q, weight_q, bias, a_s, a_b, w_s = ctx.saved_tensors
        _input, weight = ctx.other
        a_g, a_num_bit = ctx.a_param 
        w_g, w_num_bit, per_channel = ctx.w_param 
        backward_num_bit, per_sample = ctx.back_param
        
        # quantize output gradient - per_sample gradient option
        if QuantizationConfig.quantize_gradient:
            grad_output = grad_quantize_linear(grad_output, num_bit_grad=backward_num_bit, per_sample=per_sample)

        # input gradient
        grad_input = grad_output@weight_q
        
        # weight gradient - switch back option
        if QuantizationConfig.switch_back:
            grad_weight = grad_output.t()@_input
        else:
            grad_weight = grad_output.t()@_input_q
            
        # bias gradient
        if bias is not None:
            grad_bias = grad_output.sum(0)
        else:
            grad_bias = None
            
        # lsq+ param gradient
        grad_input, grad_a_s, grad_a_b = activation_quantizer_backward(grad_input, _input, a_s, a_b, a_g, a_num_bit)
        grad_weight, grad_w_s = weight_quantizer_backward(grad_weight, weight, w_s, w_g, w_num_bit, per_channel)

        return grad_input, grad_weight, grad_bias, grad_a_s, grad_a_b, None, None, grad_w_s, None, None, None, None, None

def linear_q(_input, weight, bias,
             a_s, a_b, a_g, a_num_bit, 
             w_s, w_g, w_num_bit, per_channel,
             backward_num_bit, per_sample):
    
    return LinearQ.apply(_input, weight, bias,
                         a_s, a_b, a_g, a_num_bit, 
                         w_s, w_g, w_num_bit, per_channel,
                         backward_num_bit, per_sample)

#----------------------------------------------------------------------------
# custom fqt fully-connected layer class
# forward options: lsq+, per_channel weight forward
# backward options: gradient quantization, per_sample gradient backward, switch back
@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        
        # module init
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

        # quantization params
        self.per_channel = QuantizationConfig.per_channel
        self.per_sample = QuantizationConfig.per_sample
        self.backward_num_bit = QuantizationConfig.backward_num_bit
        
        # forward quantiation (lsq plus) params initialization 
        self.a_num_bit = QuantizationConfig.activation_num_bit
        self.w_num_bit = QuantizationConfig.weight_num_bit
        self.a_s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.a_b = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        if self.per_channel:
            self.w_s = torch.nn.Parameter(torch.ones(out_features), requires_grad=True)
        else:
            self.w_s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.a_init_stat = torch.Tensor([0])
        self.w_init_stat = torch.Tensor([0])
        self.a_g = float(0)
        self.w_g = float(0)
        self.batch_init = QuantizationConfig.batch_init

    def forward(self, x):
        
        # update initialization state & quantization params
        self.a_s, self.a_b, self.a_g, self.a_init_stat = activation_quantizer_init_update(x, self.a_s, self.a_b, self.a_g, self.a_num_bit, self.a_init_stat, batch_init = self.batch_init)
        self.w_s, self.w_g, self.w_init_stat = weight_quantizer_init_update(self.weight, self.w_s, self.w_g, self.w_num_bit, self.w_init_stat, self.per_channel, batch_init = self.batch_init)
        
        # custom linear function 
        # forward: lsq+, per_channel weight forward
        # backward: gradient quantization, per_sample gradient backward, switch back
        x = linear_q(_input=x, weight=self.weight, bias=self.bias, 
                     a_s=self.a_s, a_b=self.a_b, a_g=self.a_g, a_num_bit=self.a_num_bit, 
                     w_s=self.w_s, w_g=self.w_g, w_num_bit=self.w_num_bit, per_channel=self.per_channel,
                     backward_num_bit=self.backward_num_bit, per_sample=self.per_sample)

        return x


#----------------------------------------------------------------------------
# custom linear function
# forward options: lsq+, per_channel weight forward
# backward options: gradient quantization, per_sample gradient backward, switch back
class Conv2dQ(Function):
    @staticmethod
    def forward(ctx, _input, weight, bias, groups, stride, padding,
                a_s, a_b, a_g, a_num_bit, 
                w_s, w_g, w_num_bit, per_channel,
                backward_num_bit, per_sample):
        
        # quantize input & weight - lsq+ option & per_channel weight option
        _input_q = activation_quantize(_input, a_s, a_b, a_num_bit)
        weight_q = weight_quantize(weight, w_s, w_num_bit, per_channel)
        
        # save param for backward prop
        ctx.save_for_backward(_input_q, weight_q, bias, a_s, a_b, w_s)
        ctx.other = _input, weight 
        ctx.a_param = a_g, a_num_bit 
        ctx.w_param = w_g, w_num_bit, per_channel
        ctx.back_param = backward_num_bit, per_sample
        ctx.groups, ctx.stride, ctx.padding = groups, stride, padding

        # conv2d manipulation
        output = conv2d(input=_input_q, weight=weight_q, stride=stride, padding=padding, groups=groups)
        
        # bias manipulation
        if bias is not None:
            bias = bias.to(_input_q.dtype) if bias is not None else None
            output = output.add_(bias.reshape(1, -1, 1, 1))
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # obtain saved tensors/params
        _input_q, weight_q, bias, a_s, a_b, w_s = ctx.saved_tensors
        _input, weight = ctx.other
        a_g, a_num_bit = ctx.a_param
        w_g, w_num_bit, per_channel = ctx.w_param
        backward_num_bit, per_sample = ctx.back_param
        groups, stride, padding =  ctx.groups, ctx.stride, ctx.padding
        
        # quantize gradient - per_sample gradient option
        if QuantizationConfig.quantize_gradient:
            grad_output = grad_quantize_conv(grad_output, num_bit_grad=backward_num_bit, per_sample=per_sample)

        # input gradient
        grad_input = torch.nn.grad.conv2d_input(_input_q.size(), weight_q, grad_output, groups=groups, stride=stride, padding=padding)
        
        # weight gradient - switch back option
        if QuantizationConfig.switch_back:
            grad_weight = torch.nn.grad.conv2d_weight(_input, weight_q.size(), grad_output, groups=groups, stride=stride, padding=padding)
        else:
            grad_weight = torch.nn.grad.conv2d_weight(_input_q, weight_q.size(), grad_output, groups=groups, stride=stride, padding=padding)
        
        # If bias exists, compute its gradient as well
        if bias is not None:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None
            
        # lsq+ param gradient
        grad_input, grad_a_s, grad_a_b = activation_quantizer_backward(grad_input, _input, a_s, a_b, a_g, a_num_bit)
        grad_weight, grad_w_s = weight_quantizer_backward(grad_weight, weight, w_s, w_g, w_num_bit, per_channel)

        return  grad_input, grad_weight, grad_bias, None, None, None, grad_a_s, grad_a_b, None, None,  grad_w_s, None, None, None, None, None
    

def conv2d_q(_input, weight, bias,
             a_s, a_b, a_g, a_num_bit,
             w_s, w_g, w_num_bit, per_channel,
             backward_num_bit, per_sample,
             groups=1, stride=1, padding=1):
    
    return Conv2dQ.apply(_input, weight, bias, groups, stride, padding,
                              a_s, a_b, a_g, a_num_bit, 
                              w_s, w_g, w_num_bit, per_channel,
                              backward_num_bit, per_sample)

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.
@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        
        # module init
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)
        
        if self.weight is not None:
            # quantization params
            self.per_channel = QuantizationConfig.per_channel
            self.per_sample = QuantizationConfig.per_sample
            self.backward_num_bit = QuantizationConfig.backward_num_bit
            
            # forward quantiation (lsq+) params initialization
            self.a_num_bit = QuantizationConfig.activation_num_bit
            self.w_num_bit = QuantizationConfig.weight_num_bit
            self.a_s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.a_b = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            if self.per_channel:
                self.w_s = torch.nn.Parameter(torch.ones(out_channels), requires_grad=True)
            else:
                self.w_s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.a_init_stat = torch.Tensor([0])
            self.w_init_stat = torch.Tensor([0])
            self.a_g = float(0)
            self.w_g = float(0)
            self.batch_init = QuantizationConfig.batch_init


    def forward(self, x):
        
        if self.weight is not None:
            # update initialization state & quantization params
            self.a_s, self.a_b, self.a_g, self.a_init_stat = activation_quantizer_init_update(x, self.a_s, self.a_b, self.a_g, self.a_num_bit, self.a_init_stat, batch_init = self.batch_init)
            self.w_s, self.w_g, self.w_init_stat = weight_quantizer_init_update(self.weight, self.w_s, self.w_g, self.w_num_bit, self.w_init_stat, self.per_channel, batch_init = self.batch_init)

        #filter
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = self.weight.shape[-1] // 2 if self.weight is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and self.weight is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            # x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
            x =conv2d_q(_input=x, weight=self.weight, bias=self.bias,
                        a_s=self.a_s, a_b=self.a_b, a_g=self.a_b, a_num_bit=self.a_num_bit, 
                        w_s=self.w_s, w_g=self.w_g, w_num_bit=self.w_num_bit, per_channel=self.per_channel,
                        backward_num_bit=self.backward_num_bit, per_sample=self.per_sample,
                        padding=max(w_pad - f_pad, 0),)
        elif self.fused_resample and self.down and self.weight is not None:
            # x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = conv2d_q(_input=x, weight=self.weight, bias=self.bias,
                         a_s=self.a_s, a_b=self.a_b, a_g=self.a_b, a_num_bit=self.a_num_bit, 
                         w_s=self.w_s, w_g=self.w_g, w_num_bit=self.w_num_bit, per_channel=self.per_channel,
                         backward_num_bit=self.backward_num_bit, per_sample=self.per_sample,
                         padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.weight is not None:
                # x = torch.nn.functional.conv2d(x, w, padding=w_pad)
                x = conv2d_q(_input=x, weight=self.weight, bias=self.bias,
                             a_s=self.a_s, a_b=self.a_b, a_g=self.a_b, a_num_bit=self.a_num_bit, 
                             w_s=self.w_s, w_g=self.w_g, w_num_bit=self.w_num_bit, per_channel=self.per_channel,
                             backward_num_bit=self.backward_num_bit, per_sample=self.per_sample,
                             groups=1, stride=1, padding=w_pad)
        # if b is not None:
        #     x = x.add_(b.reshape(1, -1, 1, 1))
        return x
