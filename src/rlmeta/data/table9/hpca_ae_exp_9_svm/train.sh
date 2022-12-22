# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

python train_ppo_cyclone.py train_device="cuda:0" infer_device="cuda:1" num_train_rollouts=48 num_train_workers=24 num_eval_rollouts=4 num_eval_workers=2 env_config=hpca_ae_exp_9_svm
