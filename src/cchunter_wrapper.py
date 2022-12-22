import copy

from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import gym

from autocorrelation import autocorrelation
from cache_guessing_game_env_impl import CacheGuessingGameEnv


class CCHunterWrapper(gym.Env):
    def __init__(self,
                 env_config: Dict[str, Any],
                 keep_latency: bool = True) -> None:
        env_config["cache_state_reset"] = False

        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)
        # self.cc_hunter_detection_reward = env_config.get(
        #     "cc_hunter_detection_reward", -1.0)
        self.cc_hunter_coeff = env_config.get("cc_hunter_coeff", 1.0)
        self.cc_hunter_check_length = env_config.get("cc_hunter_check_length",
                                                     4)

        self._env = CacheGuessingGameEnv(env_config)
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address

        self.step_count = 0
        self.cc_hunter_history = []

        self.no_guess = True
        self.no_guess_reward = env_config["no_guess_reward"]

    def reset(self, victim_address=-1, seed: int = -1):
        self.step_count = 0
        self.cc_hunter_history = []
        obs = self._env.reset(victim_address=victim_address,
                              reset_cache_state=True,
                              seed=seed)
        self.victim_address = self._env.victim_address
        self.no_guess = True
        return obs

    def cc_hunter_attack(self, data: Sequence[int]) -> Tuple[float, int]:
        # Mulong: only calculate 4 * size_cache size lag
        n = min(len(data), self._env.cache_size * self.cc_hunter_check_length)

        x = np.asarray(data)
        corr = [autocorrelation(x, i) for i in range(n)]
        corr = np.asarray(corr[1:])
        corr = np.nan_to_num(corr)
        mask = corr > self.threshold

        rew = -np.square(corr).mean().item() if len(corr) > 0 else 0.0
        cnt = mask.sum().item()

        return rew, cnt

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self.step_count += 1

        # is_guess = (self._env.parse_action(action)[1] == 1)
        cur_step_obs = obs[0, :]
        latency = cur_step_obs[0] if self.keep_latency else -1

        # self.cc_hunter_history.append(latency)
        # self.cc_hunter_history.append(None if latency == 2 else latency)

        # Mulong Luo
        # change the semantics of cc_hunter_history following the paper
        # only append when there is a conflict miss (i.e., victim_latency is 1(miss))
        # then check the action
        # if the action is attacker access, then it is T->S append 1
        # else if the action is trigger victim, then it is S->T append 0
        if "victim_latency" in info and info["victim_latency"] == 1:
            self.cc_hunter_history.append(0)
        elif latency == 1:
            self.cc_hunter_history.append(1)

        if info["is_guess"]:
            self.no_guess = False

        # self.cc_hunter_history.append(info.get("cache_state_change", None))

        if done:
            obs = self._env.reset(victim_address=-1,
                                  reset_cache_state=False,
                                  reset_observation=self.reset_observation)
            self.victim_address = self._env.victim_address

            if self.step_count < self.episode_length:
                done = False
            # else:
            #     rew, cnt = self.cc_hunter_attack(self.cc_hunter_history)
            #     reward += self.cc_hunter_coeff * rew
            #     info["cc_hunter_attack"] = cnt
            #
            #     if self.no_guess:
            #         reward += self.no_guess_reward

        if self.step_count >= self.episode_length:
            rew, cnt = self.cc_hunter_attack(self.cc_hunter_history)
            reward += self.cc_hunter_coeff * rew
            info["cc_hunter_attack"] = cnt
            if self.no_guess:
                reward += self.no_guess_reward
            done = True

        return obs, reward, done, info

    def seed(self, seed: int) -> None:
        self._env.seed(seed)
