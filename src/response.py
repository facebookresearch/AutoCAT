# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

class Response:
    def __init__(self, hit_list, time, data=''):
        self.hit_list = hit_list
        self.time = time
        self.data = data

    def deepen(self, time, name):
        self.hit_list[name] = False
        self.time += time
