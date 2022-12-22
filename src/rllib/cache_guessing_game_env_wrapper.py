'''
Author: Mulong Luo
Date: 2022.7.10
Usage: wrapper fucntion to solve the import issues
'''

import sys
import os
import gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append('../src')

from cache_guessing_game_env_impl import CacheGuessingGameEnv
from cchunter_wrapper import CCHunterWrapper
from cyclone_wrapper import CycloneWrapper


class CacheGuessingGameEnvWrapper(CacheGuessingGameEnv):
    pass

class CycloneWrapperWrapper(CycloneWrapper):
    pass

class CCHunterWrapperWrapper(CCHunterWrapper):
    pass