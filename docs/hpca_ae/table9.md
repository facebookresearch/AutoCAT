# Table IX: bit rate, guess accuarcy and detection rate for attacks bypassing SVM-based detector

We compare bit rate, guess accuarcy and detection rate for attacks found in Table IX


![](../../fig/table9.png)

First, go to the directory.

```
cd ${GIT_ROOT}/src/rlmeta
```
(Optional) To train a config in Table IX, use the following script:

```
$ python train_ppo_cyclone.py train_device="cuda:0" infer_device="cuda:1" num_train_rollouts=48 num_train_workers=24 num_eval_rollouts=4 num_eval_workers=2 env_config=<NAME_OF_THE_CONFIG>
```

There are 3 configs (only two need training) in Table IX, and we have ```hpca_ae_exp_9_baseline```, ```hpca_ae_exp_9_svm``` correpondingly, replace ```<NAME_OF_THE_CONFIG>``` with these.

Use ```Ctrl+C``` to interrupt the training, which will save a checkpoint in the given path ```src/rlmeta/outputs/<DATE>/<TIME>/```..

To calculate the bit rate, max autocorrelation and accuracy of these scenarios, use the following.(replace ```<NAME_OF_THE_CONFIG>``` and ```<ABSOLUTE_PATH_TO_CHECKPOINT>```) correspondingly.

```
$ python sample_cyclone.py  env_config=<NAME_OF_THE_CONFIG> checkpoint=<ABSOLUTE_PATH_TO_CHECKPOINT> env_config.window_size=164 num_episodes=1000
```

Since the training takes some time, we provide pretrained checkpoints in the following directory ```src/rlmeta/data/table9/```.



To calculate the bit rate, max autocorrelation and accuracy of RL\_SVM
```
$  python sample_cyclone.py env_config=hpca_ae_exp_9_svm checkpoint=${GIT_ROOT}/AutoCAT/src/rlmeta/data/table9/hpca_ae_exp_9_svm_new/exp1/ppo_agent-499.pth num_episodes=1000
```

which printout the following in the end
```
  info                   key          mean         std           min           max    count
------  --------------------  ------------  ----------  ------------  ------------  -------
sample        episode_length  160.00000000  0.00000000  160.00000000  160.00000000     1000
sample        episode_return   24.44600000  7.39006658  -78.00000000   32.00000000     1000
sample             num_guess   25.01800000  1.94722264   21.00000000   32.00000000     1000
sample           num_correct   24.98200000  1.95388741   21.00000000   32.00000000     1000
sample          correct_rate    0.99856189  0.00759887    0.93548387    1.00000000     1000
sample              bandwith    0.15636250  0.01217014    0.13125000    0.20000000     1000
sample        cyclone_attack    0.00500000  0.07053368    0.00000000    1.00000000     1000
sample  overall_correct_rate    0.99856104  0.00000000    0.99856104    0.99856104        1
sample      overall_bandwith    0.15636250  0.00000000    0.15636250    0.15636250        1
```

The bit rate(overall_bandwidth)=0.156, correct_rate=0.999, and detection_rate (cyclone_attack)=0.005 can be read out directly.

Similarly, to calculate the bit rate, max autocorrelation and accuracy of RL\_baseline
```
$ python sample_cyclone.py env_config=hpca_ae_exp_9_baseline checkpoint=${GIT_ROOT}AutoCAT/src/rlmeta/data/table9/hpca_ae_exp_9_baseline_new/exp1/ppo_agent-499.pth num_episodes=1000
```

To calculate the bit rate, max autocorrelation and accuracy of textbook attacker

```
$ python sample_cyclone_textbook.py 
```
