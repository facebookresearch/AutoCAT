# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import logging

from typing import Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn

import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict

import model_utils

from cache_env_wrapper import CacheEnvWrapperFactory


def batch_obs(timestep: TimeStep) -> TimeStep:
    obs, reward, terminated, truncated, info = timestep
    return TimeStep(obs.unsqueeze(0), reward, terminated, truncated, info)


def unbatch_action(action: Action) -> Action:
    act, info = action
    act.squeeze_(0)
    info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
    return Action(act, info)


def run_loop(env: Env,
             agent: PPOAgent,
             victim_addr: int = -1,
             reset_cache_state: bool = False) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0

    if victim_addr == -1:
        timestep = env.reset(reset_cache_state=reset_cache_state)
    else:
        timestep = env.reset(victim_address=victim_addr,
                             reset_cache_state=reset_cache_state)

    agent.observe_init(timestep)
    while not timestep.terminated or timestep.truncated:
        # Model server requires a batch_dim, so unsqueeze here for local runs.
        timestep = batch_obs(timestep)
        action = agent.act(timestep)
        # Unbatch the action.
        action = unbatch_action(action)

        timestep = env.step(action)
        agent.observe(action, timestep)

        episode_length += 1
        episode_return += timestep.reward

    # Only correct guess has positive reward.
    correct_rate = float(episode_return > 0.0)

    metrics = {
        "episode_length": episode_length,
        "episode_return": episode_return,
        "correct_rate": correct_rate,
    }

    return metrics


def run_loops(env: Env,
              agent: PPOAgent,
              num_episodes: int = -1,
              seed: int = 0,
              reset_cache_state: bool = False) -> StatsDict:
    # env.seed(seed)
    env.reset(seed=seed)
    metrics = StatsDict()

    if num_episodes == -1:
        start = env.env.victim_address_min
        stop = env.env.victim_address_max + 1 + int(
            env.env.allow_empty_victim_access)
        for victim_addr in range(start, stop):
            cur_metrics = run_loop(env,
                                   agent,
                                   victim_addr=victim_addr,
                                   reset_cache_state=reset_cache_state)
            metrics.extend(cur_metrics)
    else:
        for _ in range(num_episodes):
            cur_metrics = run_loop(env,
                                   agent,
                                   victim_addr=-1,
                                   reset_cache_state=reset_cache_state)
            metrics.extend(cur_metrics)

    return metrics


@hydra.main(config_path="./config", config_name="sample_attack")
def main(cfg):
    # Create env
    cfg.env_config.verbose = 1
    env_fac = CacheEnvWrapperFactory(OmegaConf.to_container(cfg.env_config))
    env = env_fac(index=0)

    # Load model
    model = model_utils.get_model(cfg.model_config, cfg.env_config.window_size,
                                  env.action_space.n, cfg.checkpoint)
    model.eval()

    # Create agent
    agent = PPOAgent(model, deterministic_policy=cfg.deterministic_policy)

    # Run loops
    metrics = run_loops(env, agent, cfg.num_episodes, cfg.seed,
                        cfg.reset_cache_state)
    logging.info("\n\n" + metrics.table(info="sample") + "\n")


if __name__ == "__main__":
    main()
