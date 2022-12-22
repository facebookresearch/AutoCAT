from typing import Dict, List, Tuple

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from models.dnn import DNNEncoder


class DNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, **kwargs) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        if len(kwargs) > 0:
            custom_model_config = kwargs
        else:
            custom_model_config = model_config["custom_model_config"]

        self.latency_dim = custom_model_config["latency_dim"]
        self.victim_acc_dim = custom_model_config["victim_acc_dim"]
        self.action_dim = custom_model_config["action_dim"]
        self.step_dim = custom_model_config["step_dim"]
        self.window_size = custom_model_config["window_size"]

        self.action_embed_dim = custom_model_config["action_embed_dim"]
        self.step_embed_dim = custom_model_config["step_embed_dim"]
        self.input_dim = (self.latency_dim + self.victim_acc_dim +
                        self.action_embed_dim + self.step_embed_dim) * self.window_size 
        self.hidden_dim = custom_model_config["hidden_dim"]
        self.output_dim = num_outputs
        self.num_blocks = custom_model_config.get("num_blocks", 1)

        self.action_embed = nn.Embedding(self.action_dim,
                                         self.action_embed_dim)
        self.step_embed = nn.Embedding(self.step_dim, self.step_embed_dim)
        self.backbone = DNNEncoder(self.input_dim, self.hidden_dim,
                                   self.hidden_dim, self.num_blocks)
        self.linear_a = nn.Linear(self.hidden_dim, self.output_dim)
        self.linear_v = nn.Linear(self.hidden_dim, 1)

        self._device = None
        self._features = None

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

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str,
                                       TensorType], state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        if self._device is None:
            self._device = next(self.parameters()).device

        obs = input_dict["obs"].to(self._device)
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
        h = self.backbone(x)
        a = self.linear_a(h)
        self._features = h

        return a, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None
        v = self.linear_v(self._features)
        return v.squeeze(1)


ModelCatalog.register_custom_model("dnn_model", DNNModel)
