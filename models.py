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
        grad_input, grad_weight = backward(grad_output, *pack)
        return grad_input, grad_weight, None, None
        
class FirstSpikeTime(Function):
    @staticmethod
    def forward(ctx, input):   
        idx = torch.arange(input.shape[2], 0, -1).unsqueeze(0).unsqueeze(0).float().cuda()
        first_spike_times = torch.argmax(idx*input, dim=2).float()
        ctx.save_for_backward(input, first_spike_times.clone())
        first_spike_times[first_spike_times==0] = input.shape[2]-1
        return first_spike_times
    
    @staticmethod
    def backward(ctx, grad_output):
        input, first_spike_times = ctx.saved_tensors
        k = F.one_hot(first_spike_times.long(), input.shape[2]).float()
        grad_input = k * grad_output.unsqueeze(-1)
        return grad_input

class SpikingLinear(nn.Module):
    def __init__(self, input_dim, output_dim, T, dt, tau_m, tau_s, mu):
        super(SpikingLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = T
        self.dt = dt
        self.tau_m = tau_m
        self.tau_s = tau_s
        
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.normal_(self.weight, mu, mu)
        
        self.forward = lambda input : WrapperFunction.apply(input, self.weight, self.manual_forward, self.manual_backward)
        
    def manual_forward(self, input):
        steps = int(self.T / self.dt)
    
        V = torch.zeros(input.shape[0], self.output_dim, steps).cuda()
        I = torch.zeros(input.shape[0], self.output_dim, steps).cuda()
        output = torch.zeros(input.shape[0], self.output_dim, steps).cuda()

        while True:
            for i in range(1, steps):
                t = i * self.dt
                V[:,:,i] = (1 - self.dt / self.tau_m) * V[:,:,i-1] + (self.dt / self.tau_m) * I[:,:,i-1]
                I[:,:,i] = (1 - self.dt / self.tau_s) * I[:,:,i-1] + F.linear(input[:,:,i-1].float(), self.weight)
                spikes = (V[:,:,i] > 1.0).float()
                output[:,:,i] = spikes
                V[:,:,i] = (1-spikes) * V[:,:,i]

            if self.training:
                is_silent = output.sum(2).min(0)[0] == 0
                self.weight.data[is_silent] = self.weight.data[is_silent] + 1e-1
                if is_silent.sum() == 0:
                    break
            else:
                break

        return (input, I, output), output
    
    def manual_backward(self, grad_output, input, I, post_spikes):
        steps = int(self.T / self.dt)
                
        lV = torch.zeros(input.shape[0], self.output_dim, steps).cuda()
        lI = torch.zeros(input.shape[0], self.output_dim, steps).cuda()
        
        grad_input = torch.zeros(input.shape[0], input.shape[1], steps).cuda()
        grad_weight = torch.zeros(input.shape[0], *self.weight.shape).cuda()
        
        for i in range(steps-2, -1, -1):
            t = i * self.dt
            delta = lV[:,:,i+1] - lI[:,:,i+1]
            grad_input[:,:,i] = F.linear(delta, self.weight.t())
            lV[:,:,i] = (1 - self.dt / self.tau_m) * lV[:,:,i+1] + post_spikes[:,:,i+1] * (lV[:,:,i+1] + grad_output[:,:,i+1]) / (I[:,:,i] - 1 + 1e-10)
            lI[:,:,i] = lI[:,:,i+1] + (self.dt / self.tau_s) * (lV[:,:,i+1] - lI[:,:,i+1])
            spike_bool = input[:,:,i].float()
            grad_weight -= (spike_bool.unsqueeze(1) * lI[:,:,i].unsqueeze(2))

        return grad_input, grad_weight
        
class SNN(nn.Module):
    def __init__(self, input_dim, output_dim, T, dt, tau_m, tau_s):
        super(SNN, self).__init__()
        self.slinear1 = SpikingLinear(input_dim, 10, T, dt, tau_m, tau_s, 0.1)
        self.outact = FirstSpikeTime.apply
        
    def forward(self, input):
        u = self.slinear1(input)
        u = self.outact(u)
        return u
        
class SpikeCELoss(nn.Module):
    def __init__(self, T, xi, tau_s):
        super(SpikeCELoss, self).__init__()
        self.xi = xi
        self.tau_s = tau_s
        self.celoss = nn.CrossEntropyLoss()
        
    def forward(self, input, target):
        loss = self.celoss(-input / (self.xi * self.tau_s), target)
        return loss
