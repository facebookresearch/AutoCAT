import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super(ResidualBlock, self).__init__()

        self.dim = dim
        layers = []
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.dim, self.dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.dim, self.dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class DNNEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_blocks: int = 1) -> None:
        super(DNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.num_blocks):
            layers.append(ResidualBlock(self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
