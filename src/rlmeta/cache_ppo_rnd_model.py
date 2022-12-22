import os
import sys

from typing import Dict, List, Tuple

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote

from rlmeta.agents.ppo.ppo_rnd_model import PPORNDModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import CacheBackbone


class CachePPORNDModel(PPORNDModel):
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
                 num_blocks: int = 1) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.policy_net = CacheBackbone(latency_dim, victim_acc_dim,
                                        action_dim, step_dim, window_size,
                                        action_embed_dim, step_embed_dim,
                                        hidden_dim, num_blocks)
        self.target_net = CacheBackbone(latency_dim, victim_acc_dim,
                                        action_dim, step_dim, window_size,
                                        action_embed_dim, step_embed_dim,
                                        hidden_dim, num_blocks)
        self.predict_net = CacheBackbone(latency_dim, victim_acc_dim,
                                         action_dim, step_dim, window_size,
                                         action_embed_dim, step_embed_dim,
                                         hidden_dim, num_blocks)
        self.linear_a = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear_ext_v = nn.Linear(self.hidden_dim, 1)
        self.linear_int_v = nn.Linear(self.hidden_dim, 1)

        self._device = None

    def forward(
            self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.policy_net(obs)
        p = self.linear_a(h)
        logpi = F.log_softmax(p, dim=-1)
        ext_v = self.linear_ext_v(h)
        int_v = self.linear_int_v(h)
        return logpi, ext_v, int_v

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            x = obs.to(self._device)
            d = deterministic_policy.to(self._device)
            logpi, ext_v, int_v = self.forward(x)

            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(d, greedy_action, sample_action)
            logpi = logpi.gather(dim=-1, index=action)

            return action.cpu(), logpi.cpu(), ext_v.cpu(), int_v.cpu()

    @remote.remote_method(batch_size=None)
    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        if self._device is None:
            self._device = next(self.parameters()).device
        reward = self._rnd_err(obs.to(self._device))
        return reward.cpu()

    def rnd_loss(self, obs: torch.Tensor) -> torch.Tensor:
        return self._rnd_err(obs).mean() * 0.5

    def _rnd_err(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.target_net(obs)
        pred = self.predict_net(obs)
        err = (pred - target).square().mean(-1, keepdim=True)
        return err
