import torch
from spikingjelly.clock_driven.surrogate import SurrogateFunctionBase, heaviside

class rectangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.vth = vth
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            mask1 = (x.abs() > ctx.vth / 2)
            mask_ = mask1.logical_not()
            grad_x = grad_output * x.masked_fill(mask_, 1./ctx.vth).masked_fill(mask1, 0.)
        return grad_x, None

class Rectangle(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rectangle.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.min(torch.max(1. / alpha * x, 0.5), -0.5)

