'''
CacheSimulatorSIMDWrapper
wraps multiple environment with different initialization into a single env
'''
#from msilib.schema import DuplicateFile
from random import random
import sys
import os
import gym
from gym import spaces

from cache_guessing_game_env_wrapper import CacheGuessingGameEnvWrapper as CacheGuessingGameEnv
#from cache_guessing_game_env_impl import *
import pdb
import sys
import signal
# random initialization
# same secret
class CacheSimulatorSIMDWrapper(gym.Env):
    def __init__(self, env_config, duplicate = 1, victim_addr = -1):
        self.duplicate = duplicate
        self.env_list = []
        self.env_config = env_config
        self.victim_addr = victim_addr 
        self.env = CacheGuessingGameEnv(env_config)
        self.victim_address_min = self.env.victim_address_min
        self.victim_address_max = self.env.victim_address_max
        self.observation_space = spaces.MultiDiscrete(list(self.env.observation_space.nvec) * self.duplicate)
        self.action_space = self.env.action_space
        self.env_list.append(CacheGuessingGameEnv(env_config))
        self.env_config['verbose'] = False
        for _ in range(1,self.duplicate):
            self.env_list.append(CacheGuessingGameEnv(env_config))
    
    def reset(self, victim_addr = -1):
        total_state = []
        # same victim_addr (secret) for all environments
        if self.victim_addr == -1 and victim_addr == -1:
            victim_addr = random.randint(self.env.victim_address_min, self.env.victim_address_max)  
        elif victim_addr == -1:
            victim_addr = self.victim_addr

        for env in self.env_list:
            state = env.reset(victim_addr)
            env._randomize_cache()#mode="union")
            total_state += list(state)
        return total_state

    def step(self, action):
        early_done_reward = 0
        total_reward = 0
        total_state = [] 
        total_done = False
        done_arr = []
        for env in self.env_list:
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_state += list(state)
            done_arr.append(done)
            if done:
                total_done = True
        
        if total_done:
            for done in done_arr:
                if done == False:
                    total_reward -= early_done_reward

        info = {}    
        return total_state, total_reward, total_done, info 


# multiple initialization
# multiple secret
class CacheSimulatorMultiGuessWrapper(gym.Env):
    def __init__(self, env_config):
        self.duplicate = 4
        self.block_duplicate = 4 
        self.env_list = []
        self.env_config = env_config
        self.env = CacheSimulatorSIMDWrapper(env_config, duplicate=self.duplicate)
        #permute the victim addresses
        self.secret_size = self.env.victim_address_max - self.env.victim_address_min + 1
        self.victim_addr_arr = [] #np.random.permutation(range(self.env.victim_address_min, self.env.victim_address_max+1))
        for _ in range(self.block_duplicate):
            #for _ in range(self.secret_size):
            rand = random.randint(self.env.victim_address_min, self.env.victim_address_max )
            self.victim_addr_arr.append(rand)
        self.observation_space = spaces.MultiDiscrete(list(self.env.observation_space.nvec) * self.block_duplicate )
        self.action_space = spaces.MultiDiscrete([self.env.action_space.n] + [self.secret_size] * self.block_duplicate)
        
        self.env_config['verbose'] = True
        self.env_list.append(CacheSimulatorSIMDWrapper(env_config, duplicate=self.duplicate, victim_addr=self.victim_addr_arr[0]))
        self.env_config['verbose'] = False
        for i in range(1, len(self.victim_addr_arr)):
        #for victim_addr in self.victim_addr_arr:
            #self.env_list.append(CacheSimulatorSIMDWrapper(env_config, duplicate=self.duplicate, victim_addr = victim_addr))
            #self.env_config['verbose'] = False
            #for _ in range(0,self.block_duplicate):
            self.env_list.append(CacheSimulatorSIMDWrapper(env_config, duplicate=self.duplicate, victim_addr=self.victim_addr_arr[i]))
    
    def reset(self):
        total_state = []
        # same victim_addr (secret) for all environments
        #self.victim_addr_arr = np.random.permutation(range(self.env.victim_address_min, self.env.victim_address_max+1))
        self.victim_addr_arr = [] #np.random.permutation(range(self.env.victim_address_min, self.env.victim_address_max+1))
        for _ in range(self.block_duplicate):
            #for _ in range(self.secret_size):
            rand = random.randint(self.env.victim_address_min, self.env.victim_address_max)
            #print('self.env.victim_address_min')
            #print(self.env.victim_address_min)
            #print('self.env.victim_address_max')
            #print(self.env.victim_address_max)
            #print('rand')
            #print(rand)
            #pdb.set_trace()
            #exit(0)
            self.victim_addr_arr.append(rand)

        for i in range(len(self.env_list)):
            env = self.env_list[i]
            #print('len(self.env_list)')
            #print(len(self.env_list))
            #print('i')
            #print(i)
            #print('victim_addr_arr')
            #print(len(self.victim_addr_arr))
            state = env.reset(self.victim_addr_arr[i])
            total_state += list(state)
        return total_state

    def step(self, action):
        early_done_reward = 0
        total_reward = 0
        total_state = [] 
        total_done = False
        done_arr = []
        orig_action = action[0] # first digit is the original action
        parsed_orig_action = self.env.env.parse_action(orig_action)
        is_guess = parsed_orig_action[1]                                      # check whether to guess or not
        is_victim = parsed_orig_action[2]                                     # check whether to invoke victim
        #is_flush = orig_action[3]                                      # check if it is a guess
        if is_victim != True and is_guess == True:
            guess_addrs = action[1:]
            for i in range(0, len(self.env_list)):
                env = self.env_list[i]
                #pdb.set_trace()
                action = orig_action  - orig_action % self.secret_size + guess_addrs[i] - self.env.env.victim_address_min           
                _, is_guesss, _, _, _ = self.env.env.parse_action(action) 
                state, reward, done, info = env.step(action)
                assert(is_guesss == True)
                assert(done == True)
                total_reward += reward
                total_state += list(state)
            info = {}            
            return total_state, total_reward * 1.0 / self.duplicate / self.block_duplicate, True, info     
        for env in self.env_list:
            state, reward, done, info = env.step(orig_action)
            total_reward += reward
            total_state += list(state)
            done_arr.append(done)
            if done:
                total_done = True
        info = {}    
        return total_state, total_reward * 1.0 / self.duplicate / self.block_duplicate , total_done, info     

if __name__ == "__main__":
    from ray.rllib.agents.ppo import PPOTrainer
    import ray
    import ray.tune as tune
    ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1)
    if ray.is_initialized():
        ray.shutdown()
    #tune.register_env("cache_guessing_game_env_fix", CacheSimulatorSIMDWrapper)#
    tune.register_env("cache_guessing_game_env_fix", CacheSimulatorMultiGuessWrapper)
    config = {
        'env': 'cache_guessing_game_env_fix', #'cache_simulator_diversity_wrapper',
        'env_config': {
            'verbose': 1,
            "force_victim_hit": False,
            'flush_inst': True,#False,
            "allow_victim_multi_access": True,#False,
            "attacker_addr_s": 0,
            "attacker_addr_e": 7,
            "victim_addr_s": 0,
            "victim_addr_e": 3,
            "reset_limit": 1,
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