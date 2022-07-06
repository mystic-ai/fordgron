import torch
import torch.nn as nn
from torch import Tensor
import math

class MLP(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        ff_dim = 4 * args["embedding_dim"]
        self.dense_h_to_4h = nn.Linear(args["embedding_dim"], ff_dim, device=device)
        self.dense_4h_to_h = nn.Linear(ff_dim, args["embedding_dim"], device=device)
        self.activation_function = NewGELUActivation()

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_function(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))