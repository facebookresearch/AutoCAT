# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+ '/third_party/cachequery/tool/')
from cachequery import CacheQuery

class CacheQueryWrapper(CacheQuery):
    pass