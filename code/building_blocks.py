import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
import numpy as np
import os
from random import *

cudnn_convolution = load(name="doing_convolution_layer_", sources=["cudnn_conv.cpp"], verbose=True)

#%%
"""
Feedback Alignment

Instead of using transposed weight of forward path,
we use weight_fa as random fixed weight for making grad_input.
The weight_fa is fixed because grad_weight_fa = 0
"""

class linear_fa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, weight_fa):
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input,  bias, weight, weight_fa)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weight,weight_fa = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_weight_fa = None
       
        grad_weight = F.linear(input.t(), grad_output.t()).t()
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
        grad_input = F.linear(grad_output, weight_fa.t())
    
        return grad_input, grad_weight, grad_bias, grad_weight_fa

class Linear_FA(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_FA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).zero_())
        else:
            self.register_parameter('bias', None)
        self.weight_fa = nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-1,1), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight)
        
    def forward(self, input):
        return linear_fa.apply(input, self.weight, self.bias, self.weight_fa)


class conv2d_fa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, weight_fa, stride=1, padding=0, groups=1):
        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.save_for_backward(input, bias, weight, weight_fa)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weight, weight_fa = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = grad_weight_fa = None
        grad_weight = cudnn_convolution.convolution_backward_weight(input, weight_fa.shape, grad_output, stride, padding, (1, 1), groups, False, False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        grad_input = cudnn_convolution.convolution_backward_input(input.shape, weight_fa, grad_output, stride, padding, (1, 1), groups, False, False)
        return grad_input, grad_weight, grad_bias, grad_weight_fa, None, None, None, None, None, None

class Conv2d_FA(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(Conv2d_FA, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        
        self.weight_fa = nn.Parameter(self.weight, requires_grad=True)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))
        
    def forward(self, input):
        return conv2d_fa.apply(input, self.weight, self.bias, self.weight_fa, self.stride, self.padding, self.groups)


#%%
""""
Direct Feedback alignment

Feedback_Receiver module receives top error and transforms the top error through random fixed weights.
First, it makes dummy data and sends it to Top_Gradient module 
which distributes top error in forward prop.
And then, top error from Top_Gradient module is transformed by weight_fb in backward prop 

Top_Gradient module sends top error to lower layers which is made by loss function.
First, it receives dummy data from layers that will receive errors in forward prop.
And then, top error is sent to the layers that gave the dummy data in backward prop.

So, the Feedback_Receiver module is located behind the layer that wants to receive the error, 
and the Top_Gradient module is located at the end of the architecture. 
The dummy created in Feedback_Receiver must be accepted in Top_Gradient.
"""
class feedback_receiver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_fb):
        output = input.clone()
        dummy = torch.Tensor(input.size()[0],weight_fb.size()[0]).zero_().to(input.device)
        ctx.save_for_backward(weight_fb,)
        ctx.shape = input.shape
        return output, dummy
    
    @staticmethod
    def backward(ctx, grad_output, grad_dummy):
        weight_fb, = ctx.saved_tensors
        input_size = ctx.shape
        grad_weight_fb = None
        
        grad_input = torch.mm(grad_dummy.view(grad_dummy.size()[0],-1), weight_fb).view(input_size) # Batch_size, input
        return grad_input, grad_weight_fb


class Feedback_Receiver(nn.Module):
    def __init__(self, connect_features):
        super(Feedback_Receiver, self).__init__()
        self.connect_features = connect_features
        self.weight_fb = None
    
    def forward(self, input):
        if self.weight_fb is None:
            self.weight_fb = nn.Parameter(torch.Tensor(self.connect_features, *input.size()[1:]).view(self.connect_features, -1)).to(input.device)
            nn.init.normal_(self.weight_fb, std = math.sqrt(1./self.connect_features))
        return feedback_receiver.apply(input, self.weight_fb)
   
class top_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, *dummies):
        output = input.clone()
        ctx.save_for_backward(output ,*dummies)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, *dummies = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_dummies = [grad_output.clone() for dummy in dummies]
        return tuple([grad_input, *grad_dummies])

class Top_Gradient(nn.Module):
    def __init__(self):
        super(Top_Gradient, self).__init__()
    
    def forward(self, input, *dummies):
        return top_gradient.apply(input, *dummies)
    
#%%
""""
Activation Sharing with Asymmetric Paths

Conv2d_FA_ASAP can do ASAP by learning weight with shared activation.
First, Inputs of Conv2d_FA_ASAP are input and shared.
The shared is determined by the activation of previous layers outside the module.

And then, like FA, the grad_input is made by weight_fa, not weight in backward prop.
However, it makes grad_weight by shared, not input for activation sharing.
Finally, grad_weight_fa = grad_weight for learning backward weight_fa.

We can use wt option. When wt = True, the grad_input is made by weight and weight_fa is no longer used.
In other words, this module only does Activation Sharing, not Activation Sharing with Asymmetric Path.
"""        
    
class conv2d_fa_asap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, shared, weight, weight_fa, bias, stride=1, padding=0, groups=1, wt = False):
        
        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        shared = shared.detach().clone()
        shared_channel_ratio = int(input.size(1)/shared.size(1)) #for matching shared activation with actual activation
        shared_filter_ratio = int(shared.size(2)/input.size(2)) #for matching shared activation with actual activation
        
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.shared_channel_ratio = shared_channel_ratio 
        ctx.shared_filter_ratio = shared_filter_ratio
        ctx.wt = wt
        ctx.save_for_backward(input, weight, weight_fa, bias, shared)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_fa, bias, shared = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        wt = ctx.wt
        
        shared_channel_ratio = ctx.shared_channel_ratio
        shared_filter_ratio = ctx.shared_filter_ratio
        grad_input = grad_weight = grad_bias  = None
        
        # Matching shared activation with actual activation by concatenation and maxpool.
        shared = torch.cat([shared] * shared_channel_ratio, 1)
        shared = F.max_pool2d(shared, shared_filter_ratio)       
        
        if wt:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is weight transport. 
            grad_weight = cudnn_convolution.convolution_backward_weight(shared, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = None
            grad_input = cudnn_convolution.convolution_backward_input(shared.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False)
        else:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is no weight transport. 
            # we traind weight_fa by grad_weight_fa = grad_weight
            grad_weight = cudnn_convolution.convolution_backward_weight(shared, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = grad_weight
            grad_input = cudnn_convolution.convolution_backward_input(shared.shape, weight_fa, grad_output, stride, padding, (1, 1), groups, False, False)
            
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        
        return grad_input, None, grad_weight, grad_weight_fa, grad_bias, None, None, None, None

class Conv2d_FA_ASAP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, wt = False):
        super(Conv2d_FA_ASAP, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.weight_fa = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.wt = wt # by using wt, Activation Saring with weight transport is possible.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))      
        
    def forward(self, input, shared):
        return conv2d_fa_asap.apply(input, shared, self.weight, self.weight_fa, self.bias, self.stride, self.padding, self.groups, self.wt) 

#%%
