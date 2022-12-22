'''
Author: Mulong Luo
Date: 2022.7.10
Function: Add one reveal action so that the agent has to explicit reveal the secret,
once the secret is revealed, it must make a guess immediately
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

import signal
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

class CacheGuessingGameWithRevealEnv(gym.Env):
    def __init__(self, env_config):

        from cache_guessing_game_env_wrapper import CacheGuessingGameEnvWrapper as CacheGuessingGameEnv
        self.env = CacheGuessingGameEnv(env_config)   

        self.action_space_size = self.env.action_space.n + 1 # increase the action space by one
        self.action_space = spaces.Discrete(self.action_space_size)
        self.observation_space = self.env.observation_space
        
        self.revealed = False # initially 

        done = False
        reward = 0 
        info = {}
        state = self.env.reset()
        self.last_unmasked_tuple = (state, reward, done, info)

    def reset(self):
        self.revealed = False # reset the revealed 
        done = False
        reward = 0 
        info = {}
        state = self.env.reset()
        self.last_unmasked_tuple = (state, reward, done, info)
        return state

    def step(self, action):
        if action == self.action_space_size - 1:
            if self.revealed == True:
                self.env.vprint("double reveal! terminated!")
                state, reward, done, info = self.last_unmasked_tuple
                reward = self.env.wrong_reward
                done = True
                return state, reward, done, info

            self.revealed = True
            self.env.vprint("reveal observation")
            # return the revealed obs, reward,# return the revealed obs, reward,  
            state, reward, done, info = self.last_unmasked_tuple
            reward = 0 # reveal action does not cost anything
            return state, reward, done, info

        elif action < self.action_space_size - 1: # this time the action must be smaller than sction_space_size -1
            _, is_guess, _, _, _ = self.env.parse_action(action)
            # need to check if revealed first
            # if revealed, must make a guess
            # if not revealed can do any thing
            if self.revealed == True:
                if is_guess == 0: # revealed but not guess # huge penalty
                    self.env.vprint("reveal but no guess! terminate")
                    done = True
                    reward = self.env.wrong_reward
                    info = {}
                    state = self.env.reset()
                    return state, reward, done, info
                elif is_guess != 0:  # this must be guess and terminate
                    return self.env.step(action)
            elif self.revealed == False:
                if is_guess != 0:
                    # guess without revewl --> huge penalty
                    self.env.vprint("guess without reward! terminate")
                    done = True
                    reward = self.env.wrong_reward
                    info = {}
                    state = self.env.reset()
                    return state, reward, done, info   
                else:
                    state, reward, done, info = self.env.step(action)
                    self.last_unmasked_tuple = ( state.copy(), reward, done, info )
                    # mask the state so that nothing is revealed
                    state[:,0] = np.zeros((state.shape[0],))
                    return state, reward, done, info


if __name__ == "__main__":
    ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1, local_mode=True)
    if ray.is_initialized():
        ray.shutdown()
    tune.register_env("cache_guessing_game_env", CacheGuessingGameWithRevealEnv)
    config = {
        'env': 'cache_guessing_game_env', #'cache_simulator_diversity_wrapper',
        'env_config': {
            'verbose': 1,
            "rerandomize_victim": False,
            "force_victim_hit": False,
            'flush_inst': False,
            "allow_victim_multi_access": True,#False,
            "allow_empty_victim_access": True,
            "attacker_addr_s": 0,
            "attacker_addr_e": 8,#4,#11,#15,
            "victim_addr_s": 0,
            "victim_addr_e": 0,#7,
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
                  "associativity": 4,  
                  "hit_time": 1, #cycles
                  "prefetcher": "nextline"
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