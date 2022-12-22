# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

#for i in `seq 0 4`; do
    echo $1
    python run_gym_rllib_agent_blacklist.py configs/config1 1 2 $1 &
    python run_gym_rllib_agent_blacklist.py configs/config1 1 4 $1 &
    python run_gym_rllib_agent_blacklist.py configs/config1 2 1 $1 &
    python run_gym_rllib_agent_blacklist.py configs/config1 4 1 $1 &
    python run_gym_rllib_agent_blacklist.py configs/config1 2 2 $1 & 
#done
