import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+ '/third_party/cachequery/tool/')
from cachequery import CacheQuery

class CacheQueryWrapper(CacheQuery):
    pass