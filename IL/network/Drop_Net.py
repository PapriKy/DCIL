import torch
import torch.nn as nn

class DropNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args, device='cpu'):
        super(DropNetwork, self).__init__()
        self.args = args
        self.device = device
        self.tanh = nn.Tanh()

        self.input_layer = nn.Linear(obs_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_depth - 1)])
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.ELU()

    def forward(self, x, mask=None):
        if mask is not None:
            weighted_input = self.input_layer.weight * mask
        else:
            weighted_input = self.input_layer.weight

        x = torch.matmul(x, weighted_input.t()) + self.input_layer.bias
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        q = self.output_layer(x)
        return q