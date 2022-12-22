# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

#!/bin/bash

LOG_PATH=$1

for f in ${LOG_PATH}/checkpoint*; do 
    for y in $f/checkpoint-*; do
            if [ -z "$(echo $y| grep -v tune)" ]; then
                echo ""
            else
                echo "python replay_checkpoint.py $(echo $y| head -1); " 
                python replay_checkpoint.py $(echo $y| grep -v tune) ; 
            fi
    done ; 
done