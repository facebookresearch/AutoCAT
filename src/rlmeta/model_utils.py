from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cache_ppo_mlp_model import CachePPOMlpModel
from cache_ppo_lstm_model import CachePPOLstmModel
from cache_ppo_transformer_model import CachePPOTransformerModel


def get_model(cfg: Dict[str, Any],
              window_size: int,
              output_dim: int,
              checkpoint: Optional[str] = None) -> nn.Module:
    cfg.args.step_dim = window_size
    if "window_size" in cfg.args:
        cfg.args.window_size = window_size
    cfg.args.output_dim = output_dim

    model = None
    if cfg.type == "mlp":
        model = CachePPOMlpModel(**cfg.args)
    elif cfg.type == "lstm":
        model = CachePPOLstmModel(**cfg.args)
    elif cfg.type == "transformer":
        model = CachePPOTransformerModel(**cfg.args)

    if model is not None and checkpoint is not None:
        params = torch.load(checkpoint)
        model.load_state_dict(params)

    return model
