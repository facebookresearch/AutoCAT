# using ray 1.92 to run
# python 3.9

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.ppo import PPOTrainer
import gym
import ray.tune as tune
from torch.nn import functional as F
from typing import Optional, Dict
import torch.nn as nn
import ray
from collections import deque
#from ray.rllib.agents.ppo.ppo_torch_policy import ValueNetworkMixin
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
#from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import Deprecated
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, sequence_mask
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    PolicyID, LocalOptimizer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import copy
import numpy as np
import sys
import math
sys.path.append("../src")
torch, nn = try_import_torch()
from cache_guessing_game_env_wrapper import CacheGuessingGameEnvWrapper as CacheGuessingGameEnv
from categorization_parser import *

def custom_init(policy: Policy, obs_space: gym.spaces.Space, 
              action_space: gym.spaces.Space, config: TrainerConfigDict)->None:
        #pass
        policy.past_len = 5        
        policy.past_models = deque(maxlen =policy.past_len)
        policy.timestep = 0

def copy_model(model: ModelV2) -> ModelV2:
    copdied_model= TorchModelV2(
        obs_space = model.obs_space,
        action_space = model.action_space, 
        num_outputs = model.num_outputs,
        model_config = model.model_config,
        name = 'copied')
    
    return copied_model

def compute_div_loss(policy: Policy, model: ModelV2,
                      dist_class: ActionDistribution,
                      train_batch: SampleBatch):
    #original_weight = copy.deepcopy(policy.get_weights())
    
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()
    valid_mask = torch.ones_like(values, dtype=torch.bool)
    dist = dist_class(logits, model)
    #log_probs = dist.logp(train_batch[SampleBatch.ACTIONS])#.reshape(-1) 
    print('log_probs')
    #print(log_probs)
    divs = []
    #div_metric = nn.KLDivLoss(size_average=False, reduce=False)
    div_metric = nn.KLDivLoss(reduction = 'batchmean')
    #div_metric = nn.CrossEntropyLoss()
    #if len(policy.past_models) > 1:
    #    assert(policy.past_models[0].state_dict() == policy.past_models[1].state_dict())
    for idx, past_model in enumerate(policy.past_models):
    #for idx, past_weights in enumerate(policy.past_weights):
        #temp_policy = pickle.loads(pickle.dumps(policy))
        #temp_policy.set_weights(past_weights)    
        #temp_model = pickle.loads(pickle.dumps(policy.model))
        #temp_model.load_state_dict(past_weights)
        #past_model.load_state_dict(policy.past_weights[i])
        #past_model = temp_model.set_weights(past_weights)    
        #assert(False)
        past_logits, _ = past_model.from_batch(train_batch)
        past_values = past_model.value_function()
        past_valid_mask = torch.ones_like(past_values, dtype=torch.bool)
        past_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], past_model)
        div = math.atan( - policy.timestep_array[idx] + policy.timestep ) * math.exp( ( policy.timestep_array[idx] - policy.timestep ) / policy.timestep_array[idx]) * dist.kl(past_dist)
        
        ###print(div)
        ###print(dist)
        ###print(past_dist)
        ###print(train_batch[SampleBatch.ACTION_DIST_INPUTS])
        #print(train_batch[SampleBatch.ACTIONS])
        #print(log_probs)
        #print(past_log_probs)
        #print(train_batch[Postprocessing.ADVANTAGES])
        #print(log_probs * train_batch[Postprocessing.ADVANTAGES])
        #print(past_log_probs * train_batch[Postprocessing.ADVANTAGES])


        #div = dist.multi_kl(past_dist) * train_batch[Postprocessing.ADVANTAGES]
        #assert(
        
        if idx == 0 and True:#policy.timestep % 10 == 0:
            print('past_model.state_dict()')
            #print(past_model.state_dict())
            print('model.state_dict()')
            #print(model.state_dict())
            #div = past_dist.multi_kl(dist)
            print('div')
            #print(div)
    
        div = div.sum().mean(0)
        divs.append(div)
    print('divs')
    #print(divs)
    div_loss = 0
    div_loss_orig = 0

    for div in divs:
        div_loss += div
        div_loss_orig += div
    
    if len(policy.past_models) > 0:
        div_loss = div_loss / len(policy.past_models)#policy.past_len
    
    print('len(policy.past_models)')
    print(len(policy.past_models))
    #policy.set_weights(original_weight)

    return div_loss


def compute_div_loss_weight(policy: Policy, weight,
                      dist_class: ActionDistribution,
                      train_batch: SampleBatch):
    original_weight = copy.deepcopy(policy.get_weights())
    policy.set_weights(weight)    
    model = policy.model
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()
    valid_mask = torch.ones_like(values, dtype=torch.bool)
    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS])#.reshape(-1) 
    print('log_probs')
    #print(log_probs)
    divs = []
    div_metric = nn.KLDivLoss(size_average=False, reduce=False)
    #div_metric = nn.CrossEntropyLoss()
    #if len(policy.past_models) > 1:
    #    assert(policy.past_models[0].state_dict() == policy.past_models[1].state_dict())
    
    for idx, past_weight in enumerate(policy.past_weights):
        #assert(False)
        policy.set_weights(past_weight)    
        past_model = policy.model
        past_logits, _ = past_model.from_batch(train_batch)
        past_values = past_model.value_function()
        past_valid_mask = torch.ones_like(past_values, dtype=torch.bool)
        past_dist = dist_class(past_logits, past_model)
        past_log_probs = past_dist.logp(train_batch[SampleBatch.ACTIONS])#.reshape(-1) 
        div =  div_metric(log_probs * train_batch[Postprocessing.ADVANTAGES], past_log_probs* train_batch[Postprocessing.ADVANTAGES])
        #div = div_metric(log_probs, past_log_probs) * train_batch[Postprocessing.ADVANTAGES]
        #div = dist.multi_kl(past_dist) * train_batch[Postprocessing.ADVANTAGES]
        #assert(
        
        if idx == 0 and True:#policy.timestep % 10 == 0:
            print('past_model.state_dict()')
            #print(past_model.state_dict())
            print('model.state_dict()')
            #print(model.state_dict())
            #div = past_dist.multi_kl(dist)
            print('div')
            #print(div)
    
        div = div.mean(0)
        divs.append(div)
    print('divs')
    #print(divs)
    div_loss = 0
    div_loss_orig = 0

    for div in divs:
        div_loss += div
        div_loss_orig += div
    
    if len(policy.past_weights) > 0:
        div_loss = div_loss / len(policy.past_weights)#policy.past_len
    
    #print('len(policy.past_weights)')
    #print(len(policy.past_weights))
    #policy.set_weights(original_weight)
    return div_loss

import pickle
def custom_loss(policy: Policy, model: ModelV2,
                      dist_class: ActionDistribution,
                      train_batch: SampleBatch) -> TensorType:
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()
    policy.timestep += 1
    #if len(policy.devices) > 1:
        # copy weights of main model (tower-0) to all other towers type
    if policy.timestep % 100 == 0:
        copied_model = pickle.loads(pickle.dumps(model))
        copied_model.load_state_dict(model.state_dict())
        policy.past_models.append(copied_model)
    
    if policy.is_recurrent():
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS],
                                  max_seq_len)
        valid_mask = torch.reshape(mask_orig, [-1])
    else:
        valid_mask = torch.ones_like(values, dtype=torch.bool)
    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
    
    #print('log_probs')
    #print(log_probs)
    
    pi_err = -torch.sum(
        torch.masked_select(log_probs * train_batch[Postprocessing.ADVANTAGES],
                            valid_mask))
    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_err = 0.5 * torch.sum(
            torch.pow(
                torch.masked_select(
                    values.reshape(-1) -
                    train_batch[Postprocessing.VALUE_TARGETS], valid_mask),
                2.0))
    # Ignore the value function.
    else:
        value_err = 0.0
    entropy = torch.sum(torch.masked_select(dist.entropy(), valid_mask))
    div_loss = compute_div_loss(policy, model, dist_class, train_batch)
    total_loss = (pi_err + value_err * policy.config["vf_loss_coeff"] -
                  entropy * policy.config["entropy_coeff"] - 1000 * div_loss )
    print('pi_err')
    #print(pi_err)
    print('value_err')
    #print(value_err)
    print('div_loss')
    print(div_loss)
    print('pi_err')
    print(pi_err)
    print('total_loss')
    print(total_loss)
    
    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = pi_err
    model.tower_stats["value_err"] = value_err
    return total_loss


CustomPolicy = A3CTorchPolicy.with_updates(
    name="MyCustomA3CTorchPolicy",
    loss_fn=custom_loss,
    #make_model= make_model,
    before_init=custom_init)
CustomTrainer = A2CTrainer.with_updates(
    get_policy_class=lambda _: CustomPolicy)
#PPOCustomPolicy = PPOTorchPolicy.with_updates(
#    name="MyCustomA3CTorchPolicy",
#    loss_fn=custom_loss,
#    #make_model= make_model,
#    before_init=custom_init)

from typing import Dict, List, Type, Union
from ray.rllib.utils.annotations import override

class CustomPPOTorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.past_len = 10
        #self.categorization_parser = CategorizationParser()        
        self.past_models = deque(maxlen =self.past_len)
        #self.past_weights = deque(maxlen= self.past_len)
        self.timestep = 0
        self.timestep_array = deque(maxlen=self.past_len)
        super(CustomPPOTorchPolicy, self).__init__(observation_space, action_space, config)

    #@override(PPOTorchPolicy)
    def loss(self, model: ModelV2, dist_class: Type[ActionDistribution],
             train_batch: SampleBatch, extern_trigger = True ) -> Union[TensorType, List[TensorType]]:
        #return custom_loss(self, model, dist_class, train_batch)
    
        self.timestep += 1
        if self.timestep % 20 == 0 and extern_trigger == False:
            copied_model = pickle.loads(pickle.dumps(model))
            copied_model.load_state_dict(model.state_dict())
            self.past_models.append(copied_model)
        
        total_loss = PPOTorchPolicy.loss(self, model, dist_class, train_batch)
        #self.past_len
        div_loss = 0 #compute_div_loss(self, model, dist_class, train_batch)
        #div_loss = compute_div_loss_weight(self, copy.deepcopy(self.get_weights()), dist_class, train_batch)
        print('total_loss')
        print(total_loss)
        print('div_loss')
        print(div_loss)
        #assert(False)
        ret_loss = total_loss - 0.03 * div_loss
        return ret_loss
        '''
        new_loss = []
        if issubclass(type(total_loss),TensorType):
            return total_loss - compute_div_loss(self, model, dist_class, train_batch)
        else:            
            for loss in total_loss:
                new_loss.append(loss - compute_div_loss(self, model, dist_class, train_batch))
            return new_loss
        '''

    def replay_agent(self, env):
        # no cache randomization
        # rangomized inference ( 10 times)
        pattern_buffer = []
        num_guess = 0
        num_correct = 0
        for victim_addr in range(env.victim_address_min, env.victim_address_max + 1):
            for repeat in range(1):
                obs = env.reset(victim_address=victim_addr)
                action_buffer = []
                done = False
                while done == False:
                    print(f"-> Sending observation {obs}")
                    action = self.compute_single_action(obs, explore=False) # randomized inference
                    print(f"<- Received response {action}")
                    obs, reward, done, info = env.step(action)
                    action_buffer.append((action, obs[0]))
                if reward > 0:
                    correct = True
                    num_correct += 1
                else:
                    correct = False
                num_guess += 1
                pattern_buffer.append((victim_addr, action_buffer, correct))
        pprint.pprint(pattern_buffer)
        return 1.0 * num_correct / num_guess, pattern_buffer        

    def push_current_model(self):
        #print('len(self.past_weights)')
        #print(len(self.past_weights))
        model = pickle.loads(pickle.dumps(self.model))
        model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.past_models.append(model)
        self.timestep_array.append(self.timestep)
        #self.past_weights.append(copy.deepcopy(self.get_weights()))
        #self.past_weights.append(copy.deepcopy(agent.get_weights()))
        return

    #TODO(Mulong): is there an standard initialization condition???
    #def is_same_agent(self, weight1, weight2, env, trainer):
    def is_same_agent(self, model1, model2, env, trainer):
        categorization_parser = CategorizationParser(env)
        original_state_dict = copy.deepcopy(self.model.state_dict())
        #original_weights = copy.deepcopy(self.get_weights())
        for victim_addr in range(env.victim_address_min, env.victim_address_max + 1):
            obs = env.reset(victim_address=victim_addr)
            #from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
            #pp = trainer.workers.local_worker().preprocessors[DEFAULT_POLICY_ID]
            #obs = pp.transform(obs)
            done = False
            #while done == False:
            #    self.model.load_state_dict(model1.state_dict())
            #    #self.set_weights(weight1)
            #    action1 = trainer.compute_single_action(obs, explore=False) # randomized inference
            #    self.model.load_state_dict(model2.state_dict())
            #    #self.set_weights(weight2)
            #    action2 = trainer.compute_single_action(obs, explore=False) # randomized inference
            #    if action1 != action2:
            #        self.model.load_state_dict(original_state_dict) 
            #        #self.set_weights(original_weights)
            #        return False
            #    else:
            #        action = action1
            #        obs, reward, done, info = env.step(action)       
            seq1 = []
            while done == False:
                self.model.load_state_dict(model1.state_dict())
                action1 = trainer.compute_single_action(obs, explore=False) # randomized inference
                seq1.append(action1)
                obs, reward, done, info = env.step(action1)      
            
            seq2 = []
            while done == False:
                self.model.load_state_dict(model2.state_dict())
                action2 = trainer.compute_single_action(obs, explore=False) # randomized inference
                seq1.append(action2)
                obs, reward, done, info = env.step(action2)

            if categorization_parser.is_same_base_pattern(seq1, seq2) == False:
                return False

        self.model.load_state_dict(original_state_dict) 
        #self.set_weights(original_weights)
        return True

    def existing_agent(self, env, trainer):
        print('existing_agent')
        current_model = pickle.loads(pickle.dumps(self.model))
        #current_weights = copy.deepcopy(self.get_weights())
        #current_model.load_state_dict(self.model.state_dict())
        for idx, past_model in enumerate(self.past_models):
        #for idx, past_weights in enumerate(self.past_weights):
            print(idx) 
            if self.is_same_agent(current_model, past_model, env, trainer):
            #if self.is_same_agent(current_weights, past_weights, env, trainer):
                return True
        return False


PPOCustomTrainer = PPOTrainer.with_updates(
    get_policy_class=lambda _: CustomPPOTorchPolicy)



import models.dnn_model 


#tune.run(CustomTrainer, config={"env": 'Frostbite-v0', "num_gpus":0})#, 'model': { 'custom_model': 'test_model' }})
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)#Fix)
# Two ways of training
# method 2b
config = {
    'env': 'cache_guessing_game_env_fix', #'cache_simulator_diversity_wrapper',

    "evaluation_num_workers": 1, 
    "evaluation_interval": 5,

    'env_config': {
        'verbose': 1,
        "force_victim_hit": False,
        'flush_inst': False,#True,
        "allow_victim_multi_access": True, #False,
        "attacker_addr_s": 0,
        "attacker_addr_e": 15,
        "victim_addr_s": 0,
        "victim_addr_e": 7,
        "reset_limit": 1,

        "length_violation_reward": -1,
        "double_victim_access_reward": -0.001,  # must be large value if not allow victim multi access
        "victim_access_reward": -0.001,
        "correct_reward": 0.02,
        "wrong_reward": -1,
        "step_reward": -0.001,

        "cache_configs": {
                # YAML config file for cache simulaton
            "architecture": {
              "word_size": 1, #bytes
              "block_size": 1, #bytes
              "write_back": True
            },
            "cache_1": {#required
              "blocks": 8, 
              "associativity": 8,  
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
    'lr': 1e-3, # decrease lr if unstable 
    #'entropy_coeff': 0.001, 
    #'num_sgd_iter': 5, 
    #'vf_loss_coeff': 1e-05, 
    'model': {
    ###    'custom_model': 'dnn_model',#'rnn', 
    ###    'custom_model_config': {
    ###        'window_size': 40, #16, #need to match
    ###        'latency_dim': 3,
    ###        'victim_acc_dim': 2,
    ###        'action_dim': 200, # need to be precise
    ###        'step_dim': 80,#40,   # need to be precise
    ###        'action_embed_dim': 32,#,8, # can be increased 32
    ###        'step_embed_dim': 6,#4, # can be increased less than 16
    ###        'hidden_dim': 32,
    ###        'num_blocks': 1
    ###    }
    }, 
    'framework': 'torch',
}

if __name__ == "__main__":
    tune.run(PPOCustomTrainer, config=config)#config={"env": 'Freeway-v0', "num_gpus":1})
