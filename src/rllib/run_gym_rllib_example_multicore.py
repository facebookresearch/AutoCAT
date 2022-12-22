'''
Author: Mulong Luo
Date: 2022.7.11
Function: An example rllib training script
'''
from random import random
import sys
import os
###sys.path.append('../src')
from ray.rllib.agents.ppo import PPOTrainer
import ray
import ray.tune as tune
import gym
from gym import spaces

from cache_guessing_game_env_wrapper import CacheGuessingGameEnvWrapper as CacheGuessingGameEnv
import signal
import numpy as np

if __name__ == "__main__":
    ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1, local_mode=True)
    if ray.is_initialized():
        ray.shutdown()
    tune.register_env("cache_guessing_game_env", CacheGuessingGameEnv)
    config = {
        'env': 'cache_guessing_game_env', #'cache_simulator_diversity_wrapper',
        'env_config': {
            'verbose': 1,
            #'super_verbose': 1,
            "rerandomize_victim": False,
            "force_victim_hit": False,
            'flush_inst': False,
            "allow_victim_multi_access": True,#False,
            "allow_empty_victim_access": False,
            "attacker_addr_s": 4,
            "attacker_addr_e": 7,#4,#11,#15,
            "victim_addr_s": 0,
            "victim_addr_e": 3,#7,
            "reset_limit": 1,
            "cache_configs": {
                # YAML config file for cache simulaton
                "architecture": {
                  "word_size": 1, #bytes
                  "block_size": 1, #bytes
                  "write_back": True
                },
                "cache_1": {#required
                  "blocks": 4,#4, 
                  "associativity": 1,  
                  "hit_time": 1, #cycles
                  "prefetcher": "nextline"
                },
                "cache_1_core_2": {#required
                  "blocks": 4,#4, 
                  "associativity": 1,  
                  "hit_time": 1, #cycles
                  "prefetcher": "nextline"
                },   
                "cache_2": {
                    "blocks": 4,
                    "associativity": 1,
                    "hit_time": 16,
                },            
                "mem": {#required
                  "hit_time": 1000 #cycles
                }
            }
        }, 
        #'gamma': 0.9, 
        'num_gpus': 1, 
        'num_workers': 1, 
        'num_envs_per_worker': 1, 
        #'entropy_coeff': 0.001, 
        #'num_sgd_iter': 5, 
        #'vf_loss_coeff': 1e-05, 
        'model': {
            #'custom_model': 'test_model',#'rnn', 
            #'max_seq_len': 20, 
            #'custom_model_config': {
            #    'cell_size': 32
            #   }
        }, 
        'framework': 'torch',
    }
    #tune.run(PPOTrainer, config=config)
    trainer = PPOTrainer(config=config)
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    while True:
        result = trainer.train() 