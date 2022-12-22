import os
import sys

from typing import Any, Dict

from rlmeta.envs.env import Env, EnvFactory
from rlmeta.envs.gym_wrapper import GymWrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_guessing_game_env_impl import CacheGuessingGameEnv
from cchunter_wrapper import CCHunterWrapper
from cyclone_wrapper import CycloneWrapper


class CacheEnvWrapperFactory(EnvFactory):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = CacheGuessingGameEnv(self.env_config)
        env = GymWrapper(env, old_step_api=True)
        return env


class CacheEnvCCHunterWrapperFactory(EnvFactory):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        # env = CacheGuessingGameEnv(self.env_config)
        env = CCHunterWrapper(self.env_config)
        env = GymWrapper(env, old_step_api=True)
        return env


class CacheEnvCycloneWrapperFactory(EnvFactory):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        # env = CacheGuessingGameEnv(self.env_config)
        env = CycloneWrapper(self.env_config)
        env = GymWrapper(env, old_step_api=True)
        return env
