import os
import sys

from typing import Dict, List, Tuple

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote

from rlmeta.agents.ppo.ppo_model import PPOModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import CacheBackbone


class CachePPOMlpModel(PPOModel):
    def __init__(self,
                 latency_dim: int,
                 victim_acc_dim: int,
                 action_dim: int,
                 step_dim: int,
                 window_size: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.backbone = CacheBackbone(latency_dim, victim_acc_dim, action_dim,
                                      step_dim, window_size, action_embed_dim,
                                      step_embed_dim, hidden_dim, num_layers)

        self.linear_a = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear_v = nn.Linear(self.hidden_dim, 1)

        self._device = None

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        p = self.linear_a(h)
        logpi = F.log_softmax(p, dim=-1)
        v = self.linear_v(h)
        return logpi, v

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            logpi, v = self.forward(obs)
            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(deterministic_policy, greedy_action,
                                 sample_action)
            logpi = logpi.gather(dim=-1, index=action)

        return action, logpi, v
