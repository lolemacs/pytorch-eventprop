import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function

class WrapperFunction(Function):
    @staticmethod
    def forward(ctx, input, params, forward, backward):
        ctx.backward = backward
        pack, output = forward(input)
        ctx.save_for_backward(*pack)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        backward = ctx.backward
        pack = ctx.saved_tensors
        grad_weights = backward(grad_output, *pack)
        return None, grad_weights, None, None

class SpikingLinear(nn.Module):
    def __init__(self, input_dim, output_dim, T, dt, tau_m, tau_s):
        super(SpikingLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = T
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s
        
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.normal_(self.weight, 0.1, 0.1)

        self.forward = lambda input : WrapperFunction.apply(input, self.weight, self.manual_forward, self.manual_backward)
        
    def manual_forward(self, input):
        steps = int(self.T / self.dt)
    
        V = torch.zeros(input.shape[0], self.output_dim).cuda()
        I = torch.zeros(steps, input.shape[0], self.output_dim).cuda()
        post_spikes = torch.zeros(steps, input.shape[0], self.output_dim).cuda()

        for i in range(steps):
            t = i * self.dt
            V = V + (self.dt / self.tau_m) * (I[i-1] - V)
            I[i] = (1 - self.dt / self.tau_s) * I[i-1] + F.linear((input == t).float(), self.weight)
            spikes = (V > 1.0).float()
            post_spikes[i] = spikes
            V = (1-spikes) * V
            
        idx = torch.arange(post_spikes.shape[0], 0, -1).unsqueeze(-1).unsqueeze(-1).float().cuda()
        first_post_spikes = torch.argmax(idx*post_spikes, dim=0).float()

        if self.training and (first_post_spikes == 0).any():
            is_silent = (first_post_spikes == 0).max(0)[0].unsqueeze(1).float()
            self.weight.data.add_(1e-1 * is_silent)
            
        first_post_spikes[first_post_spikes==0] = self.T
        return (input, I, post_spikes), first_post_spikes
    
    def manual_backward(self, grad_output, input, I, post_spikes):
        steps = int(self.T / self.dt)
    
        lV = torch.zeros(input.shape[0], self.output_dim).cuda()
        lI = torch.zeros(input.shape[0], self.output_dim).cuda()
        grad = torch.zeros_like(self.weight)

        for i in range(steps-1, -1, -1):
            t = i * self.dt
            lV = (1 - self.dt / self.tau_m) * lV 
            lV = lV + post_spikes[i] * (lV + grad_output) / (I[i] - 1 + 1e-10)
            lI = lI + (self.dt / self.tau_s) * (lV - lI)
            spike_bool = (input == t).float()
            grad -= (spike_bool.unsqueeze(1) * lI.unsqueeze(2)).sum(0)

        return grad
        
class SNN(nn.Module):
    def __init__(self, input_dim, output_dim, T, dt, tau_m, tau_s):
        super(SNN, self).__init__()
        self.slinear = SpikingLinear(input_dim, output_dim, T, dt, tau_m, tau_s)
        
    def forward(self, input):
        return self.slinear(input)
