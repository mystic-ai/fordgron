import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        ff_dim = 4 * args.hidden_size
        self.dense_h_to_4h = nn.Linear(args.hidden_size, ff_dim, device=device)
        self.dense_4h_to_h = nn.Linear(ff_dim, args.hidden_size, device=device)

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = bias_gelu_impl(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

# noinspection PyAbstractClass
class GeLUFunction(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    # bias is an optional argument
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return gelu(inputs)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        tmp = gelu_back(grad_output, inputs)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply

@torch.jit.script
def gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def gelu_back(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
            (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g