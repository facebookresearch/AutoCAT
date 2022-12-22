# author: Mulong Luo
# usage: process the json file plotted by rllib
import json
from matplotlib import pyplot as plt
import numpy as np
import sys
import math

#pathname = '/home/mulong/ray_results/PPO_cache_guessing_game_env_fix_2022-03-30_09-03-46wrptlf7f'

assert(len(sys.argv) == 2)
pathname = '/home/geunbae/ray_results/' + sys.argv[1]
pathname += '/'
filename =  pathname + '/result.json'
configname = pathname + '/params.json'

f = open(filename)
config = json.load(open(configname))
correct_reward = config['env_config']['correct_reward']
wrong_reward = config['env_config']['wrong_reward']
step_reward = config['env_config']['step_reward']

episode_reward_mean = []
episode_len_mean = []
num_steps_sampled = []
time_total_s = []

correct_rate_threshold = 0.95

data = f.readline()
while data:
    data=json.loads(data)    
    episode_reward_mean.append(data['episode_reward_mean'])
    episode_len_mean.append(data['episode_len_mean'])
    num_steps_sampled.append(data['info']['num_steps_sampled'])
    time_total_s.append(data['time_total_s'])
    data = f.readline()

f.close()

# estimating the reward based on the following funciton
# episode_reward_reward = p * correct_reward + ( 1 - p ) * wrong_reward + ( episode_len_mean - 1 ) * step_reward
# thus p = (episode_reward_mean - wrong_reward - (episode_len_mean - 1) * step_reward) / ( correct_reward - wrong_reward )
correct_rate = [] 
for i in range(0, len(episode_reward_mean)):
    p = (episode_reward_mean[i] - wrong_reward - (episode_len_mean[i] - 1) * step_reward) / ( correct_reward - wrong_reward )
    correct_rate.append(p)


# find out the coverge_time and coverge_steps
i = 0
while i < len(correct_rate):
    if correct_rate[i] > correct_rate_threshold:
        break
    i += 1
if i == len(correct_rate):
    converge_time = math.nan
    converge_steps = math.nan
else:
    converge_time = time_total_s[i]
    converge_steps = num_steps_sampled[i]

#plotting
#print(correct_rate)
#pathname = ''
plt.plot(num_steps_sampled, correct_rate)
plt.ylim(0,1)
plt.axhline(y=correct_rate_threshold, color='r', linestyle='-')
plt.xlim(left=0)
plt.xlabel('num_steps_sampled')
plt.ylabel('correct_rate')
plt.text(0, correct_rate_threshold - 0.1, 'converge_steps ='+str(converge_steps), color='r')
plt.grid(True)
plt.savefig(pathname  + 'correct_rate_steps.png')
plt.close()


plt.plot(time_total_s, correct_rate)
plt.ylim(0,1)
plt.axhline(y=correct_rate_threshold, color='r', linestyle='-')
plt.xlim(left=0)
plt.xlabel('time_total_s')
plt.ylabel('correct_rate')
plt.text(0, correct_rate_threshold - 0.1, 'converge_time ='+str(converge_time), color='r')
plt.grid(True)
plt.savefig(pathname + 'correct_rate_time.png')
plt.close()

plt.plot(num_steps_sampled, episode_len_mean)
#plt.ylim(0,1)
converge_len=np.average(np.array(episode_len_mean[len(episode_len_mean)-100::len(episode_len_mean)-1]))
plt.axhline(y=converge_len, color='r', linestyle='-')
plt.text(0, correct_rate_threshold - 0.1, 'coverge_len ='+str(converge_len), color ='r')
plt.xlim(left=0)
plt.xlabel('num_steps_sampled')
plt.ylabel('episode_len_mean')
plt.grid(True)
plt.savefig(pathname  + 'len_steps.png')
plt.close()
if converge_steps == math.nan:
    converge_len = math.nan

print(str(converge_steps)+ ' ' + str(converge_time) + ' ' + str(converge_len))


