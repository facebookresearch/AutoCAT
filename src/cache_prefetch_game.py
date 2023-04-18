# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

# Author: Mulong Luo
# date 2021.12.3
# description: environment for study RL for side channel attack
from calendar import c
from collections import deque

import numpy as np
import random
import os
import yaml, logging
import sys
import replacement_policy
from itertools import permutations
from cache_simulator import print_cache

import gym
from gym import spaces

from omegaconf.omegaconf import open_dict

from cache_simulator import *

import time

"""
Description:
  A L1 cache with total_size, num_ways 
  assume cache_line_size == 1B

Observation:
  # let's book keep all obvious information in the observation space 
     since the agent is dumb
     it is a 2D matrix
    self.observation_space = (
      [
        3,                                          #cache latency
        len(self.attacker_address_space) + 1,       # last action
        self.window_size + 2,                       #current steps
        2,                                          #whether the victim has accessed yet
      ] * self.window_size
    )

Actions:
  action is one-hot encoding
  v = | attacker_addr | ( flush_attacker_addr ) | v | victim_guess_addr | ( guess victim not access ) |

Reward:

Starting state:
  fresh cache with nolines

Episode termination:
  when the attacker make a guess
  when there is length violation
  when there is guess before victim violation
  episode terminates
"""
class CacheGuessingGameEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, env_config={
   "length_violation_reward":-10000,
   "double_victim_access_reward": -10000,
   "force_victim_hit": False,
   "victim_access_reward":-10,
   "correct_reward":200,
   "wrong_reward":-9999,
   "step_reward":-1,
   "window_size":0,
   "attacker_addr_s":4,
   "attacker_addr_e":7,
   "victim_addr_s":0,
   "victim_addr_e":3,
   "flush_inst": False,
   "allow_victim_multi_access": True,
   "verbose":0,
   "reset_limit": 1,    # specify how many reset to end an epoch?????
   "cache_configs": {
      # YAML config file for cache simulaton
      "architecture": {
        "word_size": 1, #bytes
        "block_size": 1, #bytes
        "write_back": True
      },
      "cache_1": {#required
        "blocks": 4, 
        "associativity": 1,  
        "hit_time": 1 #cycles
      },
      "mem": {#required
        "hit_time": 1000 #cycles
      }
    }
  }
):

    self.verbose = env_config["verbose"] if "verbose" in env_config else 0
    self.super_verbose = env_config["super_verbose"] if "super_verbose" in env_config else 0
    self.logger = logging.getLogger()
    self.fh = logging.FileHandler('log')
    self.sh = logging.StreamHandler()
    self.logger.addHandler(self.fh)
    self.logger.addHandler(self.sh)
    self.fh_format = logging.Formatter('%(message)s')
    self.fh.setFormatter(self.fh_format)
    self.sh.setFormatter(self.fh_format)
    self.logger.setLevel(logging.INFO)
    if "cache_configs" in env_config:
      self.configs = env_config["cache_configs"]
    else:
      self.config_file_name = os.path.dirname(os.path.abspath(__file__))+'/../configs/config_simple_L1'
      self.config_file = open(self.config_file_name)
      self.logger.info('Loading config from file ' + self.config_file_name)
      self.configs = yaml.load(self.config_file, yaml.CLoader)
    self.vprint(self.configs)
    # cahce configuration
    self.num_ways = self.configs['cache_1']['associativity'] 
    self.cache_size = self.configs['cache_1']['blocks']
    self.flush_inst = flush_inst
    self.reset_time = 0

    if "rep_policy" not in self.configs['cache_1']:
      self.configs['cache_1']['rep_policy'] = 'lru'
    
    if 'cache_1_core_2' in self.configs:
      if "rep_policy" not in self.configs['cache_1_core_2']:
        self.configs['cache_1_core_2']['rep_policy'] = 'lru'
      self.configs['cache_1_core_2']['prefetcher'] = self.prefetcher

    #with open_dict(self.configs):
    self.configs['cache_1']['prefetcher'] = self.prefetcher


    '''
    check window size
    '''
    if window_size == 0:
      #self.window_size = self.cache_size * 8 + 8 #10 
      self.window_size = self.cache_size * 4 + 8 #10 
    else:
      self.window_size = window_size
    self.feature_size = 4


    '''
    instantiate the cache
    '''
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    self.step_count = 0

    '''
    for randomized address mapping rerandomization
    '''
    if self.rerandomize_victim == True:
      addr_space = max(self.victim_address_max, self.attacker_address_max) + 1
      self.perm = [i for i in range(addr_space)]
    
    # keeping track of the victim remap length
    self.ceaser_access_count = 0
    self.mapping_func = lambda addr : addr
    # initially do a remap for the remapped cache
    self.remap()
   
    '''
    define the action space
    '''
    self.action_space = spaces.Discrete(64 + 1)

    '''
    define the observation space
    '''
    self.max_box_value = 64  #self.window_size + 2,  2 * len(self.attacker_address_space) + 1 + len(self.victim_address_space) + 1)#max(self.window_size + 2, len(self.attacker_address_space) + 1) 
    self.observation_space = spaces.Box(low=-1, high=self.max_box_value, shape=(self.window_size, 1))
    self.state = deque([[-1]] * self.window_size)

    '''
    initilizate the environment configurations
    ''' 
    self.vprint('Initializing...')
    self.l1 = self.hierarchy['cache_1']
    self.spec

    '''
    internal guessing buffer
    does not change after reset
    '''
    self.prefetch_list = [-1] * self.window_size
    self.spec_agent = SpecAgent()

  '''
  gym API: step
  this is the function that implements most of the logic
  '''
  def step(self, action):
    
    # parse the address and take action:
    if action > 0:
        prefetch_address = action - 1 
        self.l1.prefetch(prefetch_address)
        self.prefetch_list.append(prefetch_address)

    # spec agent take an access
    spec_address=self.spec_agent.act():
    self.l1.read(spec_address)
    self.access_list.append(spec_address)

    if spec_address in self.preefetch_list:
        reward = self.accurate_timely_reward
        # spec_address is labeled as used
        self.prefetch_list_used = True

    elif prefetch_address in self.accesss_list:
        reward = self.accurate_late_reward
        # prefetch_address is labeled as used
        self.prefetch_list_used = True

    elif self.prefetch_list_used[0] == False:
        reward = self.inaccurate_reward

    elif prefeth_address == -1:
        reward = self.no_prefetch_reward

    else:
        assert(False)


    return np.array(list(self.prefetch_list)), reward, done, info


  '''
  Gym API: reset the cache state
  '''
  def reset(self,
            victim_address=-1,
            reset_cache_state=False,
            reset_observation=True,
            seed = -1):
    return np.array(list(reversed(self.state)))


