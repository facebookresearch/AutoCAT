# Table VIII: bit rate, autocorrection and accuracy for attacks bypassing CChunter

We compare the attack patterns found in Table VIII and epochs need for different replacement policies.


![](../../fig/table8.png)

First, go to the directory.

```
cd ${GIT_ROOT}/src/rlmeta
```

(Optional) To train a config in Table VIII, use the following script:

```
$ python train_ppo_cchunter.py train_device="cuda:0" infer_device="cuda:1" num_train_rollouts=48 num_train_workers=24 num_eval_rollouts=4 num_eval_workers=2 env_config=<NAME_OF_THE_CONFIG>
```

There are 3 configs (only two need training) in Table VIII, and we have ```hpca_ae_exp_8_autocor```, ```hpca_ae_exp_8_baseline``` correpondingly, replace ```<NAME_OF_THE_CONFIG>``` with these.

Use ```Ctrl+C``` to interrupt the training, which will save a checkpoint in the given path ```src/rlmeta/outputs/<DATE>/<TIME>/```.

To calculate the bit rate, max autocorrelation and accuracy of these scenarios, use the following.(replace ```<NAME_OF_THE_CONFIG>``` and ```<ABSOLUTE_PATH_TO_CHECKPOINT>```) correspondingly.

```
$ python sample_cchunter.py  env_config=<NAME_OF_THE_CONFIG> checkpoint=<ABSOLUTE_PATH_TO_CHECKPOINT> env_config.window_size=164 num_episodes=1000
```

Since the training takes some time, we provide pretrained checkpoints in the following directory ```src/rlmeta/data/table8/```.



To calculate the bit rate, max autocorrelation and accuracy of RL\_autocor 
```
$ python sample_cchunter.py env_config=hpca_ae_exp_8_autocor checkpoint=${GIT_ROOT}/src/rlmeta/data/table8/hpca_ae_exp_8_autocor/ppo_agent-696.pth env_config.window_size=164 num_episodes=1000
```

which printout the following in the end
```
  info                   key          mean         std           min           max    count
------  --------------------  ------------  ----------  ------------  ------------  -------
sample        episode_length  162.00000000  0.00000000  162.00000000  162.00000000     1000
sample        episode_return   22.71700161  5.76235450  -55.48715697   30.88211980     1000
sample             num_guess   33.70700000  1.22112694   31.00000000   40.00000000     1000
sample           num_correct   33.64900000  1.21235267   30.00000000   38.00000000     1000
sample          correct_rate    0.99831903  0.00821158    0.90909091    1.00000000     1000
sample              bandwith    0.20806790  0.00753782    0.19135802    0.24691358     1000
sample          max_autocorr    0.60822789  0.10691965    0.30434783    0.99523411     1000
sample  overall_correct_rate    0.99827929  0.00000000    0.99827929    0.99827929        1
sample      overall_bandwith    0.20806790  0.00000000    0.20806790    0.20806790        1
```
The bit rate(overall_bandwidth)=0.208, correct_rate=0.998, and max_autocorr=0.608 can be read out directly.

Similarly, to calculate the bit rate, max autocorrelation and accuracy of RL\_baseline
```
$ python sample_cchunter.py env_config=hpca_ae_exp_8_baseline checkpoint=${GIT_ROOT}/src/rlmeta/data/table8/hpca_ae_exp_8_baseline/ppo_agent-429.pth env_config.window_size=164 num_episodes=1000
```

To calculate the bit rate, max autocorrelation and accuracy of textbook attacker

```
$ python sample_cchunter_textbook.py 
```
