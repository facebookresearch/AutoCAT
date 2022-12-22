import logging
import os
import sys

from typing import Dict, Optional, Sequence, Union

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

import torch
import torch.nn

import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict

import model_utils

from cache_env_wrapper import CacheEnvCCHunterWrapperFactory

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autocorrelation import autocorrelation


def batch_obs(timestep: TimeStep) -> TimeStep:
    obs, reward, terminated, truncated, info = timestep
    return TimeStep(obs.unsqueeze(0), reward, terminated, truncated, info)


def unbatch_action(action: Action) -> Action:
    act, info = action
    act.squeeze_(0)
    info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
    return Action(act, info)


def max_autocorr(data: Sequence[int], n: int) -> float:
    n = min(len(data), n)
    x = np.asarray(data)
    corr = [autocorrelation(x, i) for i in range(n)]
    corr = np.asarray(corr[1:])
    corr = np.nan_to_num(corr)
    return corr.max()


def run_loop(
        env: Env,
        agent: PPOAgent,
        victim_addr: int = -1,
        threshold: Union[float, Sequence[float]] = 0.75) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0
    num_guess = 0
    num_correct = 0

    if victim_addr == -1:
        timestep = env.reset()
    else:
        timestep = env.reset(victim_address=victim_addr)

    agent.observe_init(timestep)
    while not (timestep.terminated or timestep.truncated):
        # Model server requires a batch_dim, so unsqueeze here for local runs.
        timestep = batch_obs(timestep)
        action = agent.act(timestep)
        # Unbatch the action.
        action = unbatch_action(action)

        timestep = env.step(action)
        agent.observe(action, timestep)

        episode_length += 1
        episode_return += timestep.reward

        if "guess_correct" in timestep.info:
            num_guess += 1
            if timestep.info["guess_correct"]:
                num_correct += 1

    autocorr_n = (env.env.env._env.cache_size *
                  env.env.env.cc_hunter_check_length)
    max_ac = max_autocorr(env.env.cc_hunter_history, autocorr_n)

    if isinstance(threshold, float):
        threshold = (threshold, )
    detect = [max_ac >= t for t in threshold]

    metrics = {
        "episode_length": episode_length,
        "episode_return": episode_return,
        "num_guess": num_guess,
        "num_correct": num_correct,
        "correct_rate": num_correct / num_guess,
        "bandwith": num_guess / episode_length,
        "max_autocorr": max_ac,
    }
    for t, d in zip(threshold, detect):
        metrics[f"detect_rate-{t}"] = d

    return metrics


def run_loops(env: Env,
              agent: PPOAgent,
              num_episodes: int = -1,
              seed: int = 0,
              reset_cache_state: bool = False,
              threshold: Union[float, Sequence[float]] = 0.75) -> StatsDict:
    # env.seed(seed)
    env.reset(seed=seed)
    metrics = StatsDict()

    num_guess = 0
    num_correct = 0
    tot_length = 0

    if num_episodes == -1:
        start = env.env.victim_address_min
        stop = env.env.victim_address_max + 1 + int(
            env.env._env.allow_empty_victim_access)
        for victim_addr in range(start, stop):
            cur_metrics = run_loop(env,
                                   agent,
                                   victim_addr=victim_addr,
                                   threshold=threshold)
            num_guess += cur_metrics["num_guess"]
            num_correct += cur_metrics["num_correct"]
            tot_length += cur_metrics["episode_length"]
            metrics.extend(cur_metrics)
    else:
        for _ in range(num_episodes):
            cur_metrics = run_loop(env,
                                   agent,
                                   victim_addr=-1,
                                   threshold=threshold)
            num_guess += cur_metrics["num_guess"]
            num_correct += cur_metrics["num_correct"]
            tot_length += cur_metrics["episode_length"]
            metrics.extend(cur_metrics)

    metrics.add("overall_correct_rate", num_correct / num_guess)
    metrics.add("overall_bandwith", num_guess / tot_length)

    return metrics


@hydra.main(config_path="./config", config_name="sample_cchunter")
def main(cfg):
    # Create env
    cfg.env_config.verbose = 1
    env_fac = CacheEnvCCHunterWrapperFactory(
        OmegaConf.to_container(cfg.env_config))
    env = env_fac(index=0)

    # Load model
    model = model_utils.get_model(cfg.model_config, cfg.env_config.window_size,
                                  env.action_space.n, cfg.checkpoint)
    model.eval()

    # Create agent
    agent = PPOAgent(model, deterministic_policy=cfg.deterministic_policy)

    # Run loops
    metrics = run_loops(env,
                        agent,
                        cfg.num_episodes,
                        cfg.seed,
                        threshold=cfg.threshold)
    logging.info("\n\n" + metrics.table(info="sample") + "\n")


if __name__ == "__main__":
    main()
