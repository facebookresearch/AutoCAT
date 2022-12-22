'''
Author: Mulong Luo
Date: 2022.7.12
Usage: wrapper for cachequery that interact with the gym environment
the observation space and action space should be the same as the original autocat
'''

from collections import deque
import signal
import numpy as np
import random
import os
import yaml, logging
import sys
from itertools import permutations
import gym
from gym import spaces

import os, cmd, sys, getopt, re, subprocess, configparser
###sys.path.append('../src')
from ray.rllib.agents.ppo import PPOTrainer
import ray
import ray.tune as tune
import gym
from gym import spaces

#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+ '/third_party/cachequery/tool/')
from cache_query_wrapper import CacheQueryWrapper as CacheQuery


class CacheQueryEnv(gym.Env):
    def __init__(self, env_config):

        #sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

        '''
        instantiate the CacheQuery
        '''
        # flags
        output = None
        verbose = False
        interactive = False
        # options
        config_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+ '/third_party/cachequery/tool/cachequery.ini' # default path
        batch = None

        # config overwrite
        cacheset = None
        level = None
        cacheset='34'
        level = 'L2'      # for 4-way cache
        # read config
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            # add method for dynamic cache check
            def cache(self, prop):
                return self.get(self.get('General', 'level'), prop)
            def set_cache(self, prop, val):
                return self.set(self.get('General', 'level'), prop, val)
            setattr(configparser.ConfigParser, 'cache', cache)
            setattr(configparser.ConfigParser, 'set_cache', set_cache)
        except:
            print("[!] Error: invalid config file")
            sys.exit(1)

        # overwrite options
        if level:
            config.set('General', 'level', level)
        if cacheset:
            config.set_cache('set', cacheset)
        if output:
            config.set('General', 'log_file', output)

        # instantiate cq
        self.CQ = CacheQuery(config)
        self.cq_command = "A B C D E F G H A B"  #establish the address alphabet to number mapping

    def reset(self):
        self.revealed = False # reset the revealed 
        done = False
        reward = 0 
        info = {}
        state = self.env.reset()
        self.last_unmasked_tuple = (state, reward, done, info)

        #reset CacheQuery Command
        self.cq_command = "A B C D E F G H A B"
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
            # return the revealed obs, reward,# return the revealed obs, reward,  
            state, reward, done, info = self.last_unmasked_tuple
            reward = 0 # reveal action does not cost anything
            self.env.vprint("reveal observation")
            # when doing reveal, launch the actual cachequery
            #self.CQ.command(self.cq_command)
            answer = self.CQ.run(self.cq_command)[0]
            #print(answer)
            if answer != None:
                lat_cq = answer.split()[answer.split().index('->')+1:]
                lat_cq_cnt = len(lat_cq) - 1
                for i in range(len(state)):
                    if state[i][0] != 2 and lat_cq_cnt >= 0:
                        if int(lat_cq[lat_cq_cnt]) > 50: # hit
                            state[i][0] = 0
                        else:                            # miss
                            state[i][0] = 1
                        lat_cq_cnt -= 1
            print(state)
            return state, reward, done, info

        elif action < self.action_space_size - 1: # this time the action must be smaller than sction_space_size -1
            tmpaction = self.env.parse_action(action) 
            address = hex(tmpaction[0]+self.env.attacker_address_min)[2:]            # attacker address in attacker_address_space
            is_guess = tmpaction[1]                                                  # check whether to guess or not
            is_victim = tmpaction[2]                                                 # check whether to invoke victim
            is_flush = tmpaction[3]                                                  # check whether to flush
            victim_addr = hex(tmpaction[4] + self.env.victim_address_min)[2:]        # victim address

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
                    done = True
                    #return self.env.step(action)
                    if int(victim_addr,16) == self.env.victim_address:
                        reward = self.env.correct_reward
                    else:
                        reward = self.env.wrong_reward
                    info = {}
                    state = self.env.reset()
                    return state, reward, done, info
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
                    # append to the cq_command
                    if is_victim == True: 
                        self.cq_command += (' ' + chr(ord('A') + self.env.victim_address))
                    elif is_flush == True:
                        self.cq_command += (' ' + chr(ord('A') + int(address, 16)) + '!') 
                    else:
                        self.cq_command += (' ' + chr(ord('A') + int(address, 16)) + '?')  

                    self.last_unmasked_tuple = ( state.copy(), reward, done, info )
                    # mask the state so that nothing is revealed
                    state[:,0] = - np.ones((state.shape[0],)) # use -1 as the default (unrevealed value)

                    #print(state)
                    return state, reward, done, info

if __name__ == "__main__":
    ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1, local_mode=True)
    if ray.is_initialized():
        ray.shutdown()
    tune.register_env("cache_guessing_game_env", CacheQueryEnv)
    config = {
        'env': 'cache_guessing_game_env', #'cache_simulator_diversity_wrapper',
        'env_config': {
            'verbose': 1,
            "prefetcher": "nextline",
            "rerandomize_victim": False,
            "force_victim_hit": False,
            'flush_inst': False,
            "allow_victim_multi_access": True,#False,
            "allow_empty_victim_access": False,
            "attacker_addr_s": 0,
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
                # for L2 cache of Intel i7-6700 
                # it is a 4-way cache, this should not be changed
                "cache_1": {#required
                  "blocks": 4,#4, 
                  "associativity": 4,  
                  "hit_time": 1 #cycles
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