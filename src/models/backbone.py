import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dnn import DNNEncoder


class CacheBackbone(nn.Module):
    def __init__(self,
                 latency_dim: int,
                 victim_acc_dim: int,
                 action_dim: int,
                 step_dim: int,
                 window_size: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 hidden_dim: int,
                 num_blocks: int = 1) -> None:
        super().__init__()

        self.latency_dim = latency_dim
        self.victim_acc_dim = victim_acc_dim
        self.action_dim = action_dim
        self.step_dim = step_dim
        self.window_size = window_size

        self.action_embed_dim = action_embed_dim
        self.step_embed_dim = step_embed_dim
        self.input_dim = (self.latency_dim + self.victim_acc_dim +
                          self.action_embed_dim +
                          self.step_embed_dim) * self.window_size
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.action_embed = nn.Embedding(self.action_dim,
                                         self.action_embed_dim)
        self.step_embed = nn.Embedding(self.step_dim, self.step_embed_dim)
        self.dnn_encoder = DNNEncoder(self.input_dim, self.hidden_dim,
                                      self.hidden_dim, self.num_blocks)

    def make_one_hot(self, src: torch.Tensor,
                     num_classes: int) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = F.one_hot(src, num_classes)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def make_embedding(self, src: torch.Tensor,
                       embed: nn.Embedding) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = embed(src)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(torch.int64)
        assert obs.dim() == 3

        batch_size = obs.size(0)
        (l, v, act, step) = torch.unbind(obs, dim=-1)

        l = self.make_one_hot(l, self.latency_dim)
        v = self.make_one_hot(v, self.victim_acc_dim)
        act = self.make_embedding(act, self.action_embed)
        step = self.make_embedding(step, self.step_embed)

        x = torch.cat((l, v, act, step), dim=-1)
        x = x.view(batch_size, -1)
        y = self.dnn_encoder(x)

        return y
