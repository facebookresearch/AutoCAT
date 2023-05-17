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
