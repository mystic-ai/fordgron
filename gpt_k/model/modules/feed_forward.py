import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        ff_dim = 4 * self.embedding_dim
        self.dense_h_to_4h = nn.Linear(
            self.embedding_dim,
            ff_dim,
            device=device,
        )
        self.dense_4h_to_h = nn.Linear(
            ff_dim,
            self.embedding_dim,
            device=device,
        )

    def forward(self, X):
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(X)
        intermediate_parallel = bias_gelu_impl(
            intermediate_parallel,
            bias_parallel,
        )
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias

# @torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
# @torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
            (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g

class GeLUFunction(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    # bias is an optional argument
    def forward(ctx, inputs, bias):
        ctx.save_for_backward(inputs, bias)
        return bias_gelu(bias, inputs)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        inputs, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, inputs)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply
