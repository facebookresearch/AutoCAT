# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

#!/bin/bash
mkdir test
for s in 1 2 3 4 5
do
    for i in 0 1 2 3 4
    do
        ./LRUattack h $s >test/"data_${s}_${i}.txt"
    done
done