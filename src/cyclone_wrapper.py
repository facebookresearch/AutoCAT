# simpple SVM based detector
# based on Cyclone
# window_size = 4
# interval_size = 20
# 1 bucket

import copy

from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import gym

from sklearn import svm
from sklearn.model_selection import cross_val_score
from cache_guessing_game_env_impl import CacheGuessingGameEnv
import pickle
import os

class CycloneWrapper(gym.Env):
    def __init__(self,
                 env_config: Dict[str, Any],
                 svm_data_path='/home/mulong/cyclone_svm_data.txt',
                 keep_latency: bool = True) -> None:
        env_config["cache_state_reset"] = False

        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 160)
        #self.threshold = env_config.get("threshold", 0.8)
        path = os.getcwdb().decode('utf-8') + '/../../../svm.txt'
        self.clf = pickle.load(open(path, 'rb'))
        self.cyclone_window_size = env_config.get("cyclone_window_size", 4)
        self.cyclone_interval_size = env_config.get("cyclone_interval_size", 40)
        self.cyclone_num_buckets = env_config.get("cyclone_num_buckets", 4)
        # self.cyclone_bucket_size = self.env_config.cache_configs.cache_1.blocks / self.cyclone_num_buckets
        self.cyclone_bucket_size = self.env_config["cache_configs"]["cache_1"][
            "blocks"] / self.cyclone_num_buckets
        self.cyclone_collect_data = env_config.get("cyclone_collect_data", False)
        self.cyclone_malicious_trace = env_config.get("cyclone_malicious_trace", False)
        self.X = []
        self.Y = []

        #self.cyclone_counters = [[0]* self.cyclone_num_buckets ] * self.cyclone_window_size
        self.cyclone_counters = []
        for j in range(self.cyclone_num_buckets):
            temp =[]
            for i in range(self.cyclone_window_size):
                temp.append(0)
            self.cyclone_counters.append(temp)
        self.cyclone_coeff = env_config.get("cyclone_coeff", 1.0)

        self.cyclone_heatmap = [[], [], [], []]

        # self.cc_hunter_detection_reward = env_config.get(
        #     "cc_hunter_detection_reward", -1.0)
        #self.cc_hunter_coeff = env_config.get("cc_hunter_coeff", 1.0)
        #self.cc_hunter_check_length = env_config.get("cc_hunter_check_length",
        # 4)

        self._env = CacheGuessingGameEnv(env_config)
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address

        self.svm_data_path = svm_data_path
        self.cnt = 0
        self.step_count = 0
        #self.cc_hunter_history = []

        self.no_guess = True
        self.no_guess_reward = env_config["no_guess_reward"]

    def load_svm_model(self):
        from numpy import loadtxt
        data = loadtxt('all.txt.svm.txt')
        X = data[:,1:]
        Y = data[:,0]
        clf = svm.SVC(random_state=0)
        clf.fit(X, Y)

    def set_victim(self, victim_addr):
        self._env.victim_address = victim_addr

    def save_svm_data(self):
        fp = open(self.svm_data_path, 'a')
        for i in range(len(self.X)):
            str1 = ' '.join(str(e) for e in self.X[i])
            str1 = str(self.Y[i]) + ' ' + str1 + '\n'
            fp.write(str1)
        fp.close()

    # if save_data==True, sa
    def reset(self,
              victim_address=-1,
              save_data=False,
              set_victim=False,
              seed: int = -1):
        if save_data == True:
            self.save_svm_data()

            # drwa figure
            print(self.cyclone_heatmap)
            #p=sns.heatmap(self.cyclone_heatmap, vmin=0, vmax=20)
            #p.set_xlabel('Time intervals (40 cycles)')
            #p.set_ylabel('Set index')
            #fig= p.get_figure()
            #fig.set_size_inches(3, 3)
            #fig_path ='/home/mulong/RL_SCA/src/CacheSimulator/src/heatmap.png'
            ##fig_path = os.getcwdb().decode('utf-8') + '/../heatmap.png'
            #fig.savefig(fig_path)

        if set_victim == True and victim_address != -1:
            obs = self._env.reset(victim_address=victim_address,
                                  reset_cache_state=False,
                                  seed=seed)
            return obs

        # reset cyclone counter
        #self.cyclone_counters = [[0]* self.cyclone_num_buckets ] * self.cyclone_window_size
        self.cyclone_counters = []
        for j in range(self.cyclone_num_buckets):
            temp =[]
            for i in range(self.cyclone_window_size):
                temp.append(0)
            self.cyclone_counters.append(temp)

        self.step_count = 0
        self.cnt = 0
        #self.cc_hunter_history = []
        obs = self._env.reset(victim_address=victim_address,
                              reset_cache_state=True,
                              seed=seed)
        self.victim_address = self._env.victim_address
        self.no_guess = True

        return obs

    ####def autocorr(self, x: np.ndarray, p: int) -> float:
    ####    if p == 0:
    ####        return 1.0
    ####    mean = x.mean()
    ####    var = x.var()
    ####    return ((x[:-p] - mean) * (x[p:] - mean)).mean() / var

    def cyclone_attack(self, cyclone_counters):
        # collect data to train svm
        #print(cyclone_counters)

        for i in range(len(cyclone_counters)):
            self.cyclone_heatmap[i] += cyclone_counters[i]

        if self.cyclone_collect_data == True:
            x = np.array(cyclone_counters).reshape(-1)
            if self.cyclone_malicious_trace == True:
                y = 1
            else:
                y = 0
            self.X.append(x)
            self.Y.append(y)
        x = np.array(cyclone_counters).reshape(-1)
        #print(x)
        ######print(x)
        ######x_mod = np.array(cyclone_counters).reshape(-1)
        ######x_mod[0] = 0
        ######y = 1
        ######y_mod = 0
        ######X = [x, x_mod]
        ######Y= [y, y_mod]
        ######clf = svm.SVC(random_state=0)
        ######clf.fit(X,Y)
        y = self.clf.predict([x])[0]
        rew = -y

        return rew.item()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        if info["is_guess"]:
            self.no_guess = False

        # is_guess = (self._env.parse_action(action)[1] == 1)
        cur_step_obs = obs[0, :]
        latency = cur_step_obs[0] if self.keep_latency else -1

        # self.cc_hunter_history.append(latency)
        # self.cc_hunter_history.append(None if latency == 2 else latency)

        # Mulong Luo
        # cyclone
        if "cyclic_set_index" in info and info["cyclic_set_index"] != -1:
            set = int(info["cyclic_set_index"])
            if self.step_count < self.episode_length:
                self.cyclone_counters[int(set / self.cyclone_bucket_size) ][int(self.step_count / self.cyclone_interval_size) ] += 1

        self.step_count += 1
        # self.cc_hunter_history.append(info.get("cache_state_change", None))

        if done:
            self.cnt += 1 #TODO(Mulong) fix the logic so taht only guess increment the cnt
            obs = self._env.reset(victim_address=-1,
                                  reset_cache_state=False,
                                  reset_observation=self.reset_observation)
            self.victim_address = self._env.victim_address

            if self.step_count < self.episode_length:
                done = False
            # else:
            #     #rew, cnt = self.cc_hunter_attack(self.cc_hunter_history)
            #     rew = self.cyclone_attack(self.cyclone_counters)
            #     reward += self.cyclone_coeff * rew
            #     info["cyclone_attack"] = (rew != 0.0) #self.cnt
            #
            #     if self.no_guess:
            #         reward += self.no_guess_reward

        if self.step_count >= self.episode_length:
            rew = self.cyclone_attack(self.cyclone_counters)
            reward += self.cyclone_coeff * rew
            info["cyclone_attack"] = (rew != 0.0)  # self.cnt
            if self.no_guess:
                reward += self.no_guess_reward
            done = True

        return obs, reward, done, info

    def seed(self, seed: int) -> None:
        self._env.seed(seed)
