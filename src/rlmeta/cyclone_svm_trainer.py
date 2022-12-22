# Author: Mulong Luo
# date: 2022.6.28
# usage: to train the svm classifier of cycloen by feeding 
# the date from TextbookAgent as malicious traces 
# and spec traces for benign traces

import logging

from typing import Dict

import hydra
import torch
import torch.nn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append("/home/mulong/RL_SCA/src/CacheSimulator/src")

import rlmeta.utils.nested_utils as nested_utils
import numpy as np
from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.types import Action
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict

from textbook_attacker import TextbookAgent
# from cache_guessing_game_env_impl import CacheGuessingGameEnv
# from cchunter_wrapper import CCHunterWrapper
from cache_env_wrapper import CacheEnvWrapperFactory, CacheEnvCycloneWrapperFactory 
from cyclone_wrapper import CycloneWrapper


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    checkpoint = trainer.save()
    print("checkpoint saved at", checkpoint)
    sys.exit(0)

class SpecAgent():
    def __init__(self, env_config, trace_file):
        self.local_step = 0
        self.lat = []
        self.no_prime = False # set to true after first prime
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] 
            self.cache_size = self.configs['cache_1']['blocks']
            attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
            assert(self.num_ways == 1) # currently only support direct-map cache
            assert(flush_inst == False) # do not allow flush instruction
            assert(attacker_addr_e - attacker_addr_s == victim_addr_e - victim_addr_s ) # address space must be shared
            #must be no shared address space
            assert( ( attacker_addr_e + 1 == victim_addr_s ) or ( victim_addr_e + 1 == attacker_addr_s ) )
            assert(self.allow_empty_victim_access == False)

        self.trace_file = trace_file
        # load the data SPEC bengin traces
        self.fp = open(self.trace_file)
        line = self.fp.readline().split()
        self.domain_id_0 = line[0]
        self.domain_id_1 = line[0]
        line = self.fp.readline().split()
        while line != '':
            self.domain_id_1 = line[0]
            if self.domain_id_1 != self.domain_id_0:
                break
            line = self.fp.readline().split()
 
        self.fp.close()
        self.fp = open(self.trace_file)
    
    def act(self, timestep):
        info = {}
        line = self.fp.readline().split()
        if len(line) == 0:
            action = self.cache_size
            addr = 0#addr % self.cache_size
            info={"file_done" : True}
            return action, info

        domain_id = line[0]
        cache_line_size = 8
        addr = int( int(line[3], 16) / cache_line_size )
        
        print(addr)
        
        if domain_id == self.domain_id_0: # attacker access
            action = addr % self.cache_size
            info ={}
        else: # domain_id = self.domain_id_1: # victim access
            action = self.cache_size
            addr = addr % self.cache_size
            info={"reset_victim_addr": True, "victim_addr": addr}
        return action, info

@hydra.main(config_path="./config", config_name="sample_cyclone")
def main(cfg):
    repeat = 80000
    trace_file = '/home/mulong/remix3.txt'
    svm_data_path = 'autocat.svm.txt' #trace_file + '.svm.txt'
    #create env
    cfg.env_config['verbose'] = 1

    # generate dataset for malicious traces
    cfg.env_config['cyclone_collect_data'] = True
    cfg.env_config['cyclone_malicious_trace'] = True
    env_fac = CacheEnvCycloneWrapperFactory(cfg.env_config)
    env = env_fac(index=0)
    env.svm_data_path = svm_data_path
    fp = open(svm_data_path,'w')
    fp.close()
    agent = TextbookAgent(cfg.env_config) 
    episode_length = 0
    episode_return = 0.0

    for i in range(repeat):
        timestep = env.reset()
        num_guess = 0
        num_correct = 0
        while not timestep.done:
            # Model server requires a batch_dim, so unsqueeze here for local runs.
            timestep.observation.unsqueeze_(0)
            action, info = agent.act(timestep)
            action = Action(action, info)
            # unbatch the action
            victim_addr = env._env.victim_address
            timestep = env.step(action)
            obs, reward, done, info = timestep
            if "guess_correct" in info:
                num_guess += 1
                if info["guess_correct"]:
                    print(f"victim_address! {victim_addr} correct guess! {info['guess_correct']}")
                    num_correct += 1
                else:
                    correct = False

            agent.observe(action, timestep)
            episode_length += 1
            episode_return += timestep.reward


    env.reset(save_data=True) # save data to file
    # generate benign traces
'''
    cfg.env_config['cyclone_collect_data'] = True
    cfg.env_config['cyclone_malicious_trace'] = False
    env_fac = CacheEnvCycloneWrapperFactory(cfg.env_config)
    env = env_fac(index=0)
    print("mix.txt opened!")

    agent = SpecAgent(cfg.env_config, trace_file)
    episode_length = 0
    episode_return = 0.0
    
    file_done = False
    # generate dataset for benign traces
    iter = 0
    while not file_done:
    #for i in range(repeat):
        timestep = env.reset()
        num_guess = 0
        num_correct = 0
        done = False
        count = 0
        iter += 1
        while not done:
            # Model server requires a batch_dim, so unsqueeze here for local runs.
            timestep.observation.unsqueeze_(0)
            action, info = agent.act(timestep)
            if "file_done" in info:
                file_done = True
                break
            if "victim_addr"  in info:
                print(info["victim_addr"])
                #env.set_victim(info["victim_addr"])
                env._env.set_victim(info["victim_addr"])
                action = Action(action, info)
            else:
                action = Action(action, info)
            # unbatch the action
            victim_addr = env._env.victim_address
            timestep = env.step(action)
            obs, reward, done, info = timestep            
            count += 1
            #if count % 10 == 0:
                #action = Action(agent.cache_size * 2, {})
                #timestep = env.step(action)
                #obs, reward, done, info = timestep            


            if count == 160:
                action = Action(agent.cache_size * 2, {})
                timestep = env.step(action)
                obs, reward, done, info = timestep    
                done = True
                count = 0
            #if "guess_correct" in info:
            #    num_guess += 1
            #    if info["guess_correct"]:
            #        print(f"victim_address! {victim_addr} correct guess! {info['guess_correct']}")
            #        num_correct += 1
            #    else:
            #        correct = False

            #agent.observe(action, timestep)
            episode_length += 1
            episode_return += timestep.reward

    env.reset(save_data=True) # save data to file
'''
    #cfg.env_config['cyclone_malicious_trace'] = False
    #env_fac = CacheEnvCCHunterWrapperFactory(cfg.env_config)
    #env = env_fac(index=0)
 

if __name__ == "__main__":
    main()
