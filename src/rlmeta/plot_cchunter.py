# script for plotting figure on paper

import logging

from typing import Dict

#import hydra
#import torch
#import torch.nn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append("/home/mulong/RL_SCA/src/CacheSimulator/src")

#import rlmeta.utils.nested_utils as nested_utils
import numpy as np
#from rlmeta.agents.ppo.ppo_agent import PPOAgent
#from rlmeta.core.types import Action
#from rlmeta.envs.env import Env
#from rlmeta.utils.stats_dict import StatsDict
#from cchunter_wrapper import CCHunterWrapper
#from cache_env_wrapper import CacheEnvWrapperFactory
#from cache_ppo_model import CachePPOModel
#from cache_ppo_transformer_model import CachePPOTransformerModel
#from textbook_attacker import TextbookAgent
# from cache_guessing_game_env_impl import CacheGuessingGameEnv
# from cchunter_wrapper import CCHunterWrapper
#from cache_env_wrapper import CacheEnvWrapperFactory
#from cache_ppo_model import CachePPOModel
#from cache_ppo_transformer_model import CachePPOTransformerModel
#from cache_ppo_transformer_periodic_model import CachePPOTransformerPeriodicModel
import matplotlib.pyplot as plt
import pandas as pd
#from cache_env_wrapper import CacheEnvCCHunterWrapperFactory
import matplotlib.font_manager as font_manager

from autocorrelation import autocorrelation

fontaxes = {
    'family': 'Arial',
     #   'color':  'black',
        'weight': 'bold',
        #'size': 6,
}
fontaxes_title = {
    'family': 'Arial',
 #       'color':  'black',
        'weight': 'bold',
      #  'size': 9,
}

font = font_manager.FontProperties(family='Arial',
                                   weight='bold',
                                   style='normal')


def autocorrelation_plot_forked(series, ax=None, n_lags=None, change_deno=False, change_core=False, **kwds):
    """
    Autocorrelation plot for time series.
    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    n_lags: maximum number of lags to show. Default is len(series)
    kwds : keywords
        Options to pass to matplotlib plotting method
    Returns:
    -----------
    class:`matplotlib.axis.Axes`
    """
    import matplotlib.pyplot as plt
    
    n_full = len(series)
    if n_full <= 2:
      raise ValueError("""len(series) = %i but should be > 2
      to maintain at least 2 points of intersection when autocorrelating
      with lags"""%n_full)
      
    # Calculate the maximum number of lags permissible
    # Subtract 2 to keep at least 2 points of intersection,
    # otherwise pandas.Series.autocorr will throw a warning about insufficient
    # degrees of freedom
    n_maxlags = n_full #- 2
    
    # calculate the actual number of lags
    if n_lags is None:
      # Choosing a reasonable number of lags varies between datasets,
      # but if the data longer than 200 points, limit this to 100 lags as a
      # reasonable default for plotting when n_lags is not specified
      n_lags = min(n_maxlags, 100)
    else:
      if n_lags > n_maxlags:
        raise ValueError("n_lags should be < %i (i.e. len(series)-2)"%n_maxlags)
    
    if ax is None:
        ax = plt.gca(xlim=(0, n_lags), ylim=(-1.1, 1.6))

    if not change_core:
      data = np.asarray(series)
      def r(h: int) -> float:
        return autocorrelation(data, h)
    else:
      def r(h):
        return series.autocorr(lag=h)
      
    # x = np.arange(n_lags) + 1
    x = np.arange(n_lags)
    # y = lmap(r, x)
    y = np.array([r(xi) for xi in x])
    print(y)
    print(f"y = {y}")
    print(f"y_max = {np.max(y[1:])}")

    z95 = 1.959963984540054
    z99 = 2.5758293035489004

    # ax.axhline(y=-z95 / np.sqrt(n_full), color='grey')
    # ax.axhline(y=-z99 / np.sqrt(n_full), linestyle='--', color='grey')
    ax.set_xlabel("Lag (p)", fontdict = fontaxes)
    ax.set_ylabel("Autocorrelation \n Coefficient", fontdict = fontaxes)
    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    return ax


def main():
    plt.figure(num=None, figsize=(5, 2), dpi=300, facecolor='w')
    series_human = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    #series_baseline = [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # sampled from python sample_cchunter.py checkpoint=/home/ml2558/CacheSimulator/src/rlmeta/data/table8/hpca_ae_exp_8_baseline_new/exp1/ppo_agent-499.pth  num_episodes=1
    series_baseline = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    #series_l2 = [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    # sampled from python sample_cchunter.py checkpoint=/home/ml2558/CacheSimulator/src/rlmeta/data/table8/hpca_ae_exp_8_autocor_new/exp1/ppo_agent-499.pth  num_episodes=1
    series_l2 = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    #series_l2 = [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]

    for i in range(0, len(series_baseline)):
      series_baseline[i] += 1.2

    for i in range(0, len(series_l2)):
      series_l2[i] += 2.4

    series_human = series_human[0:50]
    series_baseline = series_baseline[0:50]
    series_l2 = series_l2[0:50]
    ax = plt.subplot(121)
    ax.set_xlim([0, 48] )
    ax.set_ylim([-0.1, 3.7])
    ax.set_yticks([])
    plt.tick_params(left=False)
    text_x = -10
    ax.text(text_x, 0.15, 'A->V', fontproperties=font)
    ax.text(text_x, 0.85, 'V->A', fontproperties=font)  
    ax.text(text_x, 0.15+1.2, 'A->V',fontproperties=font)
    ax.text(text_x, 0.85+1.2, 'V->A',fontproperties=font)  
    ax.text(text_x, 0.15+2.4, 'A->V', fontproperties=font)
    ax.text(text_x, 0.85+2.4, 'V->A',fontproperties=font)   
    #ax.set_xlim([0, 60])
    ax.plot(series_human)#, linewidth=4 )
    ax.plot(series_baseline)
    ax.plot(series_l2)
    ax.set_xlabel("Number of cache conflicts", fontdict = fontaxes)
    ax.legend(prop={'size': 6, 'family': 'Arial', 'weight':'bold'})
    ax.legend(['textbook', 'RL_baseline', 'RL_autocor'], ncol=3,bbox_to_anchor=(2.2,1.28), prop=font)


    data_human = pd.Series(series_human)
    data_baseline = pd.Series(series_baseline)
    data_l2 = pd.Series(series_l2)
    cache_size = 4

    #plt.figure(num=None, figsize=(5.2, 2), dpi=300, facecolor='w')
    #plt.subplots_adjust(right = 0.98, top =0.97, bottom=0.24,left=0.13,wspace=0, hspace=0.2)  
    ax = plt.subplot(122)
    autocorrelation_plot_forked(data_human,ax=ax, n_lags= 8 * cache_size, change_deno=True) #consider removing -2
    autocorrelation_plot_forked(data_baseline, ax=ax,n_lags= 8 * cache_size, change_deno=True) #consider removing -2
    autocorrelation_plot_forked(data_l2, ax=ax, n_lags= 8 * cache_size, change_deno=True) #consider removing -2
    #plt.legend(['textbook', 'RL_baseline', 'RL_autocor'], ncol=3, prop=font)
    plt.plot([0,40],[0.75,0.75], linestyle='--', color='grey')
    # ax.axhline(y=z95 / np.sqrt(n_full), color='grey')
    plt.plot([0,40],[0,0], color='black')
    ax.set_xlim([0, 32] )
    ax.yaxis.set_label_coords(-0.09, .5)
    #plt.savefig('cchunter_compare.pdf')
    #plt.savefig('cchunter_compare.png')
    plt.subplots_adjust(right = 0.999, top =0.85, bottom=0.22,left=0.085,wspace=0.28, hspace=0.2)  
    plt.savefig('event_train.pdf')
    plt.savefig('event_train.png')

if __name__ == "__main__":
    main()


'''
human
 Reset...(also the cache state) 
 victim address  3 
 Step... 
 acceee 4 miss 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:0.79 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 3 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 acceee 6 miss 
 Step... 
 access 7 hit 
 Step... 
 correct guess 2 
 correct rate:0.8 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 2 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:0.81 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 1 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:0.82 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 3 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:0.83 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 0 
 correct rate:0.84 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 0 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 acceee 6 miss 
 Step... 
 access 7 hit 
 Step... 
 correct guess 2 
 correct rate:0.85 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 2 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:0.86 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 1 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 acceee 6 miss 
 Step... 
 access 7 hit 
 Step... 
 correct guess 2 
 correct rate:0.87 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 2 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:0.88 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 1 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:0.89 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 0 
 correct rate:0.9 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 0 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 0 
 correct rate:0.91 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 0 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:0.92 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 3 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 acceee 6 miss 
 Step... 
 access 7 hit 
 Step... 
 correct guess 2 
 correct rate:0.93 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 2 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:0.94 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 0 
 correct rate:0.95 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 0 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 acceee 6 miss 
 Step... 
 access 7 hit 
 Step... 
 correct guess 2 
 correct rate:0.96 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 2 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:0.97 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 1 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:0.98 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 0 
 correct rate:0.99 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 0 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 1 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 acceee 6 miss 
 Step... 
 access 7 hit 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 2 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 3 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 acceee 6 miss 
 Step... 
 access 7 hit 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 2 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 0 correct guess! True
Episode number of guess: 26
Episode number of corrects: 26
correct rate: 1.0
bandwidth rate: 0.1625
[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
/home/mulong/RL_SCA/src/CacheSimulator/src/rlmeta/sample_cchunter.py:75: MatplotlibDeprecationWarning: Calling gca() with keyword arguments was deprecated in Matplotlib 3.4. Starting two minor releases later, gca() will take no keyword arguments. The gca() function should only be used to get the current axes, or if no axes exist, create new axes with default keyword arguments. To create a new axes with non-default arguments, use plt.axes() or plt.subplot().
  ax = plt.gca(xlim=(1, n_lags), ylim=(-1.0, 1.0))
y = [ 1.         -0.98113208  0.96223727 -0.94339623  0.92447455 -0.90566038
  0.88671182 -0.86792453  0.84894909 -0.83018868  0.81118637 -0.79245283
  0.77342364 -0.75471698  0.73566091 -0.71698113  0.69789819 -0.67924528
  0.66013546 -0.64150943  0.62237274 -0.60377358  0.58461001 -0.56603774
  0.54684728 -0.52830189  0.50908456 -0.49056604  0.47132183 -0.45283019
  0.4335591  -0.41509434]
y_max = 0.9622372735580283
Figure saved as 'cchunter_hit_trace_3_acf.png
Total number of guess: 104
Total number of corrects: 104
Episode total: 640
correct rate: 1.0
bandwidth rate: 0.1625
'''

'''
l2
 Reset...(also the cache state) 
 victim address  3 
 Step... 
 victim access 3  
 Step... 
 acceee 5 miss 
 Step... 
 acceee 4 miss 
 Step... 
 acceee 6 miss 
 Step... 
 victim access 3  
 Step... 
 access 5 hit 
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 0 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 5 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 2 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 5 hit 
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 0 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 0 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 4 hit 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 3 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 4 hit 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 0 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 0 correct guess! True
 Step... 
 victim access 1  
 Step... 
 acceee 5 miss 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 1 correct guess! True
 Step... 
 victim access 1  
 Step... 
 acceee 5 miss 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 1 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 4 hit 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 3 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 access 4 hit 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 0 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 0 correct guess! True
 Step... 
 acceee 7 miss 
 Step... 
 victim access 1  
 Step... 
 acceee 5 miss 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 1 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 5 hit 
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 0 correct guess! True
 Step... 
 victim access 1  
 Step... 
 acceee 5 miss 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 0 correct guess! True
 Step... 
 victim access 1  
 Step... 
 acceee 5 miss 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 5 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 0 correct guess! True
 Step... 
 acceee 7 miss 
 Step... 
 victim access 1  
 Step... 
 access 6 hit 
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 1 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 5 miss 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 1 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 6 hit 
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 6 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 0 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 6 hit 
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 access 6 hit 
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 0 correct guess! True
 Step... 
 victim access 1  
 Step... 
 acceee 7 miss 
 Step... 
 victim access 1  
 Step... 
 victim access 1  
 Step... 
 access 6 hit 
 Step... 
 access 4 hit 
 Step... 
 acceee 5 miss 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 1 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 6 hit 
 Step... 
 access 4 hit 
 Step... 
 access 5 hit 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 7 miss 
 Step... 
 victim access 0  
 Step... 
 victim access 0  
 Step... 
 access 6 hit 
 Step... 
 acceee 4 miss 
 Step... 
 victim access 0  
 Step... 
 victim access 0  
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 access 4 hit 
 Step... 
 access 4 hit 
 Step... 
 access 4 hit 
 Step... 
 access 4 hit 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 0 correct guess! True
Episode number of guess: 32
Episode number of corrects: 32
correct rate: 1.0
bandwidth rate: 0.19753086419753085
[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
/home/mulong/RL_SCA/src/CacheSimulator/src/rlmeta/sample_cchunter.py:75: MatplotlibDeprecationWarning: Calling gca() with keyword arguments was deprecated in Matplotlib 3.4. Starting two minor releases later, gca() will take no keyword arguments. The gca() function should only be used to get the current axes, or if no axes exist, create new axes with default keyword arguments. To create a new axes with non-default arguments, use plt.axes() or plt.subplot().
  ax = plt.gca(xlim=(1, n_lags), ylim=(-1.0, 1.0))
y = [ 1.         -0.6823596   0.42214715 -0.34085761  0.25558463 -0.17101461
  0.0498516  -0.0011716  -0.01648051 -0.03524565  0.08539014 -0.13729204
  0.1872608  -0.30731079  0.42507615 -0.40935718  0.35528782 -0.30748653
  0.25324143 -0.13764352  0.01525033  0.03219948  0.01689057 -0.00187456
 -0.01718347 -0.06820667  0.08468718 -0.10228072  0.11858549 -0.1040967
  0.08451144 -0.13817074]
y_max = 0.42507615402640003
Figure saved as 'cchunter_hit_trace_3_acf.png
Total number of guess: 134
Total number of corrects: 134
Episode total: 648
correct rate: 1.0
bandwidth rate: 0.20679012345679013
'''

'''
baseline
 Reset...(also the cache state) 
 victim address  3 
 Step... 
 acceee 4 miss 
 Step... 
 acceee 7 miss 
 Step... 
 acceee 6 miss 
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 3 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 0 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 1 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 2 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 0 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 3 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 2 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 0 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 2 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 3 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 access 7 hit 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 0 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 access 6 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 0 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 0 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 7 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 2 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 access 7 hit 
 Step... 
 access 6 hit 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 0 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 7 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  1 
victim_address! 2 correct guess! True
 Step... 
 victim access 1  
 Step... 
 access 4 hit 
 Step... 
 access 7 hit 
 Step... 
 access 6 hit 
 Step... 
 correct guess 1 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 1 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 0 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 0 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 7 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 2 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 7 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 2 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 0 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 3 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 3 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 0 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 7 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 2 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 7 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 2 correct guess! True
 Step... 
 victim access 0  
 Step... 
 acceee 4 miss 
 Step... 
 correct guess 0 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 0 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  3 
victim_address! 3 correct guess! True
 Step... 
 victim access 3  
 Step... 
 access 4 hit 
 Step... 
 acceee 7 miss 
 Step... 
 correct guess 3 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  2 
victim_address! 3 correct guess! True
 Step... 
 victim access 2  
 Step... 
 access 4 hit 
 Step... 
 access 7 hit 
 Step... 
 acceee 6 miss 
 Step... 
 correct guess 2 
 correct rate:1.0 
 Reset...(cache state the same) 
 victim address  0 
victim_address! 2 correct guess! True
Episode number of guess: 38
Episode number of corrects: 38
correct rate: 1.0
bandwidth rate: 0.2360248447204969
[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
/home/mulong/RL_SCA/src/CacheSimulator/src/rlmeta/sample_cchunter.py:75: MatplotlibDeprecationWarning: Calling gca() with keyword arguments was deprecated in Matplotlib 3.4. Starting two minor releases later, gca() will take no keyword arguments. The gca() function should only be used to get the current axes, or if no axes exist, create new axes with default keyword arguments. To create a new axes with non-default arguments, use plt.axes() or plt.subplot().
  ax = plt.gca(xlim=(1, n_lags), ylim=(-1.0, 1.0))
y = [ 1.         -0.92995169  0.94312692 -0.92874396  0.91403162 -0.89975845
  0.88493632 -0.87077295  0.85584102 -0.84178744  0.82674572 -0.81280193
  0.79765042 -0.78381643  0.76855512 -0.75483092  0.73945982 -0.72584541
  0.71036451 -0.6968599   0.68126921 -0.6678744   0.65217391 -0.63888889
  0.62307861 -0.60990338  0.59398331 -0.58091787  0.56488801 -0.55193237
  0.53579271 -0.52294686]
y_max = 0.9431269213877909
Figure saved as 'cchunter_hit_trace_3_acf.png
Total number of guess: 147
Total number of corrects: 147
Episode total: 643
correct rate: 1.0
bandwidth rate: 0.2286158631415241
'''
