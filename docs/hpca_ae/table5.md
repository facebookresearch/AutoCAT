# Compare efforts of training with different replacement policies


We compare the attack patterns found in Table V and epochs need for different replacement policies.


![](../../fig/table5.png)

First, go to the directory.

```
cd ${GIT_ROOT}/src/rlmeta
```

(Optional) To train a config in Table V, use the following script:

```
$ python train_ppo_attack.py env_config=<NAME_OF_THE_CONFIG> 
```

which will print out the following:

```
/home/ml2558/miniconda3/envs/rllib/lib/python3.9/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'ppo_attack': Defaults list is missing `_self_`. See https://hydra.cc/docs/upgrades/1.0_to_1.1/default_composition_order for more information
 warnings.warn(msg, UserWarning)
workding_dir = /home/ml2558/Downloads/AutoCAT/src/rlmeta/outputs/2022-10-31/19-06-02
...
```
Please take notes of the ```working_dir```, which is the place where all the checkpoint and logs corresponding to this training is saved.

There are 3 configs in Table V, and we have ```hpca_ae_exp_5_lru```, ```hpca_ae_exp_5_plru```, ..., ```hpca_ae_exp_5_rrip``` correpondingly, replace ```<NAME_OF_THE_CONFIG>``` with these.

Use ```Ctrl+C``` to interrupt the training, which will save a checkpoint in the given path ```src/rlmeta/output/<DATE>/<TIME>/ppo_agent-<X>.pth``` and corresponding training logs in ```src/rlmeta/output/<DATE>/<TIME>/train_ppo_attack.log```. 

To extract the attack pattern from the checkpoint, use the following command (replace ```<NAME_OF_THE_CONFIG>``` and ```<ABSOLUTE_PATH_TO_CHECKPOINT>```) correspondingly.

```
$ python sample_attack.py  env_config=<NAME_OF_THE_CONFIG> checkpoint=<ABSOLUTE_PATH_TO_CHECKPOINT>
```

Since the training takes some time, we provide pretrained checkpoints in the following directory ```src/rlmeta/data/table5```. For each replacement policy, we ran the experiments three times and the results are put in ```exp_1```, ```exp_2```, ```exp_3``` folder. For each folder, we include the ```ppo_agent-X.pth``` which is the last checkpoint in the training processes, we also include the training logs.

To reproduce the attack sequence in the Table  for LRU, use the following command.

```
$ python sample_attack.py  env_config=hpca_ae_exp_5_lru checkpoint=${GIT_ROOT}/src/rlmeta/data/table5/exp_1/hpca_ae_exp_5_lru/ppo_agent-19.pth
```
To reproduce the attack sequence in the Table  for PLRU, use the following command.
```
$ python sample_attack.py  env_config=hpca_ae_exp_5_plru checkpoint=${GIT_ROOT}/src/rlmeta/data/table5/exp_1/hpca_ae_exp_5_plru/ppo_agent-24.pth
```
To reproduce the attack sequence in the Table  for RRIP, use the following command.
```
$ python sample_attack.py  env_config=hpca_ae_exp_5_rrip checkpoint=${GIT_ROOT}/src/rlmeta/data/table5/exp_1/hpca_ae_exp_5_rrip/ppo_agent-117.pth
```

The training log includes the information of epochs to coverge (i.e., when ```correct_rate>0.97```) and the episode length. For example, in ```exp_1``` of ```LRU```, look at the ```src/rlmeta/data/table5/exp_1/train_ppo_attack.log``` file, whose each row looks like the following:

```
[2022-10-30 17:07:23,695][root][INFO] - {"return": {"mean": -0.3072997082863001, "std": 0.2757334560751737, "min": -1.537475824356079, "max": 0.07956413179636002, "count": 3000, "key": "return"}, "policy_ratio": {"mean": 0.9998782961964625, "std": 0.00823920690603589, "min": 0.9599696397781372, "max": 1.0417852401733398, "count": 3000, "key": "policy_ratio"}, "policy_loss": {"mean": -0.007201023951017596, "std": 0.007786379648485109, "min": -0.03927534073591232, "max": 0.032539889216423035, "count": 3000, "key": "policy_loss"}, "value_loss": {"mean": 3.117961673557761, "std": 7.972199116957483, "min": 0.6606887578964233, "max": 75.79322052001953, "count": 3000, "key": "value_loss"}, "entropy": {"mean": 1.950882093469298, "std": 0.05904087166219686, "min": 1.8283371925354004, "max": 2.0616185665130615, "count": 3000, "key": "entropy"}, "loss": {"mean": 1.512762174715595, "std": 3.987634525140007, "min": 0.2697336971759796, "max": 37.850555419921875, "count": 3000, "key": "loss"}, "grad_norm": {"mean": 1.3915719778339057, "std": 1.7964732559132808, "min": 0.26180699467658997, "max": 19.371002197265625, "count": 3000, "key": "grad_norm"}, "sample_data_time/ms": {"mean": 0.047321661375463064, "std": 0.22424096029590987, "min": 0.024685636162757874, "max": 11.692143976688385, "count": 3000, "key": "sample_data_time/ms"}, "batch_learn_time/ms": {"mean": 23.559486703015857, "std": 11.324141836887115, "min": 22.02333603054285, "max": 641.8017027899623, "count": 3000, "key": "batch_learn_time/ms"}, "episode_length": {"mean": 5.6794365365841735, "std": 4.150082276437064, "min": 1.0, "max": 41.0, "count": 88098, "key": "episode_length"}, "episode_return": {"mean": -0.3551782106290726, "std": 0.9436530486421526, "min": -1.3800000000000001, "max": 0.99, "count": 88098, "key": "episode_return"}, "episode_time/s": {"mean": 0.046500535157658535, "std": 0.04548654639413301, "min": 0.004196850582957268, "max": 2.087366731837392, "count": 88098, "key": "episode_time/s"}, "steps_per_second": {"mean": 126.11621075150374, "std": 11.489212514234161, "min": 0.5018509572697972, "max": 238.27391045581618, "count": 88098, "key": "steps_per_second"}, "correct_rate": {"mean": 0.3458080773683819, "std": 0.47563100297937977, "min": 0.0, "max": 1.0, "count": 88098, "key": "correct_rate"}, "info": "T Epoch 0", "phase": "Train", "epoch": 0, "time": 97.51816826313734}
```
Each row represents the stats of training in one RL epoch (=3000 steps)

For example, run the following script will print the logs shown below.

```
python data/show_log.py --log_file=data/table5/hpca_ae_exp_5_lru/exp_1/train_ppo_attack.log
```

You can see that at epoch **14** which is the **15th** epoch, the training ```correct_rate``` jump to 0.99093645 which is over 0.97. And at this epoch, the evaluation ```episode_length``` is 7.0. For LRU, PLRU, and RRIP, we calculate the average of evaluation ```episode_length``` and epochs to converge among three experiments.


```
Experiment Configs = {'env_config': {'length_violation_reward': -2.0, 'double_victim_access_reward': -0.01, 'victim_access_reward': -0.01, 'correct_reward': 1.0, 'wrong_reward': -1.0, 'step_reward': -0.01, 'verbose': 0, 'force_victim_hit': False, 'flush_inst': False, 'allow_victim_multi_access': True, 'allow_empty_victim_access': True, 'attacker_addr_s': 0, 'attacker_addr_e': 4, 'victim_addr_s': 0, 'victim_addr_e': 0, 'reset_limit': 1, 'cache_configs': {'architecture': {'word_size': 1, 'block_size': 1, 'write_back': True}, 'cache_1': {'blocks': 4, 'associativity': 4, 'hit_time': 1, 'rep_policy': 'lru', 'prefetcher': 'none'}, 'mem': {'hit_time': 1000}}, 'window_size': 64}, 'model_config': {'type': 'transformer', 'args': {'latency_dim': 3, 'victim_acc_dim': 2, 'action_dim': 64, 'step_dim': 64, 'action_embed_dim': 16, 'step_embed_dim': 4, 'hidden_dim': 128, 'output_dim': -1, 'num_layers': 1}}, 'm_server_name': 'm_server', 'm_server_addr': '127.0.0.1:4411', 'r_server_name': 'r_server', 'r_server_addr': '127.0.0.1:4412', 'c_server_name': 'c_server', 'c_server_addr': '127.0.0.1:4413', 'train_device': 'cuda:0', 'infer_device': 'cuda:1', 'num_train_rollouts': 48, 'num_train_workers': 24, 'num_eval_rollouts': 4, 'num_eval_workers': 2, 'replay_buffer_size': 131072, 'prefetch': 2, 'batch_size': 512, 'optimizer': {'name': 'Adam', 'args': {'lr': 0.0001}}, 'learning_starts': 65536, 'model_push_period': 10, 'entropy_coeff': 0.02, 'num_epochs': 1000, 'steps_per_epoch': 3000, 'num_eval_episodes': 100, 'train_seed': 123, 'eval_seed': 456, 'table_view': False}

     info                  key          mean          std          min           max    count
---------  -------------------  ------------  -----------  -----------  ------------  -------
T Epoch 0               return   -0.30729971   0.27573346  -1.53747582    0.07956413     3000
T Epoch 0         policy_ratio    0.99987830   0.00823921   0.95996964    1.04178524     3000
T Epoch 0          policy_loss   -0.00720102   0.00778638  -0.03927534    0.03253989     3000
T Epoch 0           value_loss    3.11796167   7.97219912   0.66068876   75.79322052     3000
T Epoch 0              entropy    1.95088209   0.05904087   1.82833719    2.06161857     3000
T Epoch 0                 loss    1.51276217   3.98763453   0.26973370   37.85055542     3000
T Epoch 0            grad_norm    1.39157198   1.79647326   0.26180699   19.37100220     3000
T Epoch 0  sample_data_time/ms    0.04732166   0.22424096   0.02468564   11.69214398     3000
T Epoch 0  batch_learn_time/ms   23.55948670  11.32414184  22.02333603  641.80170279     3000
T Epoch 0       episode_length    5.67943654   4.15008228   1.00000000   41.00000000    88098
T Epoch 0       episode_return   -0.35517821   0.94365305  -1.38000000    0.99000000    88098
T Epoch 0       episode_time/s    0.04650054   0.04548655   0.00419685    2.08736673    88098
T Epoch 0     steps_per_second  126.11621075  11.48921251   0.50185096  238.27391046    88098
T Epoch 0         correct_rate    0.34580808   0.47563100   0.00000000    1.00000000    88098
T Epoch 0                 time   97.51816826   0.00000000  97.51816826   97.51816826        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 0    episode_length   64.00000000  0.00000000   64.00000000   64.00000000      100
E Epoch 0    episode_return   -2.63000000  0.00000000   -2.63000000   -2.63000000      100
E Epoch 0    episode_time/s    0.51980119  0.00430815    0.50764884    0.53036778      100
E Epoch 0  steps_per_second  123.13248075  1.02366465  120.67098011  126.07140085      100
E Epoch 0              time  113.54100838  0.00000000  113.54100838  113.54100838        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 1               return   -0.03490658  0.03916032   -0.17405675    0.08937775     3000
T Epoch 1         policy_ratio    0.99987417  0.00456451    0.98421389    1.01786685     3000
T Epoch 1          policy_loss   -0.00030820  0.00434812   -0.01509478    0.01673610     3000
T Epoch 1           value_loss    0.79299974  0.01387887    0.74070948    0.83703357     3000
T Epoch 1              entropy    1.83898087  0.02100459    1.75983691    1.90648270     3000
T Epoch 1                 loss    0.35941205  0.00826987    0.32930529    0.38453460     3000
T Epoch 1            grad_norm    0.73194137  0.23152727    0.20998183    1.89481902     3000
T Epoch 1  sample_data_time/ms    0.04172424  0.01739191    0.02433173    0.51289611     3000
T Epoch 1  batch_learn_time/ms   23.34930480  0.86975794   21.86381631   30.45030218     3000
T Epoch 1       episode_length    5.34484461  3.18354174    1.00000000   43.00000000    99265
T Epoch 1       episode_return   -0.05113494  1.00033610   -1.42000000    0.99000000    99265
T Epoch 1       episode_time/s    0.04305687  0.02614712    0.00483878    0.35568733    99265
T Epoch 1     steps_per_second  125.36544195  7.53635337   68.17228561  206.66356225    99265
T Epoch 1         correct_rate    0.49615675  0.49998523    0.00000000    1.00000000    99265
T Epoch 1                 time  187.32316920  0.00000000  187.32316920  187.32316920        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 1    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 1    episode_return    0.05000000  0.99819838   -1.01000000    0.99000000      100
E Epoch 1    episode_time/s    0.01567550  0.00057201    0.01435515    0.01699663      100
E Epoch 1  steps_per_second  127.75736876  4.65614399  117.67038201  139.32277573      100
E Epoch 1      correct_rate    0.53000000  0.49909919    0.00000000    1.00000000      100
E Epoch 1              time  190.33415426  0.00000000  190.33415426  190.33415426        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 2               return   -0.02522624  0.04123281   -0.16988353    0.12799752     3000
T Epoch 2         policy_ratio    0.99998839  0.00415576    0.98684931    1.01542640     3000
T Epoch 2          policy_loss   -0.00004605  0.00404890   -0.01358792    0.01519387     3000
T Epoch 2           value_loss    0.80590128  0.01229092    0.76052690    0.84820896     3000
T Epoch 2              entropy    1.81072176  0.01970617    1.73481321    1.86872602     3000
T Epoch 2                 loss    0.36669015  0.00737374    0.34057319    0.39642736     3000
T Epoch 2            grad_norm    0.69395860  0.21248833    0.20232861    1.87777638     3000
T Epoch 2  sample_data_time/ms    0.04421250  0.06736872    0.02350844    2.24100798     3000
T Epoch 2  batch_learn_time/ms   23.38185684  0.87354773   22.00584393   29.59091775     3000
T Epoch 2       episode_length    4.84327285  2.81373408    1.00000000   43.00000000    93851
T Epoch 2       episode_return   -0.03475669  1.00027302   -1.36000000    0.99000000    93851
T Epoch 2       episode_time/s    0.03898890  0.02309965    0.00490045    0.35647683    93851
T Epoch 2     steps_per_second  125.43393856  7.93925982   78.50044361  204.06278055    93851
T Epoch 2         correct_rate    0.50183802  0.49999662    0.00000000    1.00000000    93851
T Epoch 2                 time  264.26171132  0.00000000  264.26171132  264.26171132        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 2    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 2    episode_return    0.05000000  0.99819838   -1.01000000    0.99000000      100
E Epoch 2    episode_time/s    0.01581916  0.00066757    0.01409749    0.01733956      100
E Epoch 2  steps_per_second  126.65729826  5.41949175  115.34321019  141.86926973      100
E Epoch 2      correct_rate    0.53000000  0.49909919    0.00000000    1.00000000      100
E Epoch 2              time  267.26733774  0.00000000  267.26733774  267.26733774        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 3               return   -0.01869272  0.03976949   -0.15653548    0.10463151     3000
T Epoch 3         policy_ratio    0.99995050  0.00393488    0.98488188    1.01523566     3000
T Epoch 3          policy_loss   -0.00011076  0.00392920   -0.01363289    0.01605034     3000
T Epoch 3           value_loss    0.82622001  0.01130635    0.78121722    0.86237198     3000
T Epoch 3              entropy    1.77434692  0.02380567    1.67440021    1.84304571     3000
T Epoch 3                 loss    0.37751230  0.00688355    0.35345033    0.39911368     3000
T Epoch 3            grad_norm    0.67622305  0.19775014    0.23314430    1.61422467     3000
T Epoch 3  sample_data_time/ms    0.04579398  0.09810288    0.02409238    3.16428579     3000
T Epoch 3  batch_learn_time/ms   23.39479325  0.88727590   21.84233069   29.82081473     3000
T Epoch 3       episode_length    4.41741818  2.40496547    1.00000000   30.00000000   102904
T Epoch 3       episode_return   -0.03013158  1.00028134   -1.29000000    0.99000000   102904
T Epoch 3       episode_time/s    0.03552064  0.01975422    0.00536872    0.24820008   102904
T Epoch 3     steps_per_second  125.53734856  8.06776452   81.88799064  186.26401484   102904
T Epoch 3         correct_rate    0.50202130  0.49999591    0.00000000    1.00000000   102904
T Epoch 3                 time  341.19303299  0.00000000  341.19303299  341.19303299        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 3    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 3    episode_return    0.05000000  0.99819838   -1.01000000    0.99000000      100
E Epoch 3    episode_time/s    0.01536275  0.00068568    0.01396749    0.01829186      100
E Epoch 3  steps_per_second  130.43581548  5.63504627  109.33827504  143.18962284      100
E Epoch 3      correct_rate    0.53000000  0.49909919    0.00000000    1.00000000      100
E Epoch 3              time  345.20238080  0.00000000  345.20238080  345.20238080        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 4               return   -0.02091533  0.04042105   -0.16838126    0.10875066     3000
T Epoch 4         policy_ratio    1.00000836  0.00390549    0.98348713    1.01604319     3000
T Epoch 4          policy_loss   -0.00026242  0.00392054   -0.01444615    0.01422323     3000
T Epoch 4           value_loss    0.83474915  0.00918178    0.80373013    0.86624098     3000
T Epoch 4              entropy    1.75927636  0.02023688    1.67219543    1.82651567     3000
T Epoch 4                 loss    0.38192663  0.00591841    0.36023211    0.40250656     3000
T Epoch 4            grad_norm    0.65474352  0.19090218    0.22699203    1.86945474     3000
T Epoch 4  sample_data_time/ms    0.04219897  0.01619229    0.02405420    0.52701402     3000
T Epoch 4  batch_learn_time/ms   23.39246024  0.86876348   22.05115929   29.35517579     3000
T Epoch 4       episode_length    4.27971567  2.26546743    1.00000000   29.00000000   107620
T Epoch 4       episode_return   -0.03157062  1.00023409   -1.28000000    0.99000000   107620
T Epoch 4       episode_time/s    0.03441408  0.01864356    0.00588187    0.23265486   107620
T Epoch 4     steps_per_second  125.54303034  8.12969496   80.98028286  170.01384568   107620
T Epoch 4         correct_rate    0.50061327  0.49999962    0.00000000    1.00000000   107620
T Epoch 4                 time  419.16313774  0.00000000  419.16313774  419.16313774        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 4    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 4    episode_return   -0.13000000  0.99277389   -1.01000000    0.99000000      100
E Epoch 4    episode_time/s    0.01530764  0.00061391    0.01358608    0.01753452      100
E Epoch 4  steps_per_second  130.86215574  5.21240608  114.06071826  147.20944955      100
E Epoch 4      correct_rate    0.44000000  0.49638695    0.00000000    1.00000000      100
E Epoch 4              time  423.17178576  0.00000000  423.17178576  423.17178576        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 5               return   -0.01914051  0.04072691   -0.16293430    0.14621428     3000
T Epoch 5         policy_ratio    1.00002549  0.00389727    0.98505032    1.01661873     3000
T Epoch 5          policy_loss   -0.00028182  0.00388770   -0.01563659    0.01724335     3000
T Epoch 5           value_loss    0.83470848  0.00920764    0.80236989    0.86582911     3000
T Epoch 5              entropy    1.74637152  0.01954965    1.67943025    1.81018019     3000
T Epoch 5                 loss    0.38214499  0.00588348    0.36000195    0.40279096     3000
T Epoch 5            grad_norm    0.64958320  0.19236512    0.21595123    1.66976094     3000
T Epoch 5  sample_data_time/ms    0.04491389  0.07084684    0.02410263    1.90121494     3000
T Epoch 5  batch_learn_time/ms   23.38558673  0.87970685   22.15026505   30.11751827     3000
T Epoch 5       episode_length    4.21292293  2.26795231    1.00000000   40.00000000   109387
T Epoch 5       episode_return   -0.03188240  1.00026779   -1.38000000    0.99000000   109387
T Epoch 5       episode_time/s    0.03382293  0.01864556    0.00622313    0.32815168   109387
T Epoch 5     steps_per_second  125.77811326  8.12840201   80.92212997  166.90235101   109387
T Epoch 5         correct_rate    0.50012342  0.49999998    0.00000000    1.00000000   109387
T Epoch 5                 time  497.05774740  0.00000000  497.05774740  497.05774740        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 5    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 5    episode_return    0.03000000  0.99919968   -1.01000000    0.99000000      100
E Epoch 5    episode_time/s    0.01495936  0.00063467    0.01365729    0.01666082      100
E Epoch 5  steps_per_second  133.93275397  5.59749600  120.04210578  146.44192165      100
E Epoch 5      correct_rate    0.52000000  0.49959984    0.00000000    1.00000000      100
E Epoch 5              time  500.06346153  0.00000000  500.06346153  500.06346153        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 6               return   -0.02207199  0.04066799   -0.14959508    0.11506648     3000
T Epoch 6         policy_ratio    0.99997009  0.00382461    0.98466158    1.01437640     3000
T Epoch 6          policy_loss   -0.00052274  0.00383759   -0.01572716    0.01233528     3000
T Epoch 6           value_loss    0.83958744  0.00945158    0.80275512    0.87661648     3000
T Epoch 6              entropy    1.71872018  0.02393773    1.64500904    1.80324471     3000
T Epoch 6                 loss    0.38489658  0.00602870    0.36395997    0.40874854     3000
T Epoch 6            grad_norm    0.64045858  0.18183336    0.22113773    1.60751200     3000
T Epoch 6  sample_data_time/ms    0.04259647  0.03167012    0.02294965    1.66056771     3000
T Epoch 6  batch_learn_time/ms   23.45583874  0.93629967   22.15717174   29.64522783     3000
T Epoch 6       episode_length    4.05792992  2.16995010    1.00000000   52.00000000   112498
T Epoch 6       episode_return   -0.02901483  1.00024620   -1.51000000    0.99000000   112498
T Epoch 6       episode_time/s    0.03256222  0.01782433    0.00605417    0.42395677   112498
T Epoch 6     steps_per_second  125.82205780  8.28833646   76.12614370  167.13446368   112498
T Epoch 6         correct_rate    0.50078224  0.49999939    0.00000000    1.00000000   112498
T Epoch 6                 time  574.20654418  0.00000000  574.20654418  574.20654418        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 6    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 6    episode_return    0.07000000  0.99679486   -1.01000000    0.99000000      100
E Epoch 6    episode_time/s    0.01541356  0.00081317    0.01287687    0.01776601      100
E Epoch 6  steps_per_second  130.11405333  6.81872854  112.57452899  155.31724579      100
E Epoch 6      correct_rate    0.54000000  0.49839743    0.00000000    1.00000000      100
E Epoch 6              time  578.21499934  0.00000000  578.21499934  578.21499934        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 7               return   -0.02282922  0.04017967   -0.16807516    0.11757468     3000
T Epoch 7         policy_ratio    0.99979379  0.00386362    0.98510373    1.01578617     3000
T Epoch 7          policy_loss   -0.00059640  0.00379203   -0.01316835    0.01495202     3000
T Epoch 7           value_loss    0.84737253  0.00855258    0.81573558    0.87538338     3000
T Epoch 7              entropy    1.69407873  0.02188763    1.62485170    1.75220084     3000
T Epoch 7                 loss    0.38920829  0.00555715    0.36727473    0.40723526     3000
T Epoch 7            grad_norm    0.63787664  0.18513413    0.19699834    1.50653362     3000
T Epoch 7  sample_data_time/ms    0.04305948  0.04906974    0.02512056    1.96114928     3000
T Epoch 7  batch_learn_time/ms   23.46397940  0.98279747   22.15079684   31.18773270     3000
T Epoch 7       episode_length    3.93320544  2.04865724    1.00000000   56.00000000   117285
T Epoch 7       episode_return   -0.03060246  1.00026869   -1.55000000    0.99000000   117285
T Epoch 7       episode_time/s    0.03162106  0.01688850    0.00638913    0.45277306   117285
T Epoch 7     steps_per_second  125.59540005  8.35898328   84.49219599  168.60788871   117285
T Epoch 7         correct_rate    0.49936480  0.49999960    0.00000000    1.00000000   117285
T Epoch 7                 time  652.34537546  0.00000000  652.34537546  652.34537546        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 7    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 7    episode_return    0.11000000  0.99277389   -1.01000000    0.99000000      100
E Epoch 7    episode_time/s    0.01532979  0.00052553    0.01427388    0.01682010      100
E Epoch 7  steps_per_second  130.61784586  4.46397546  118.90536949  140.11606253      100
E Epoch 7      correct_rate    0.56000000  0.49638695    0.00000000    1.00000000      100
E Epoch 7              time  656.35739026  0.00000000  656.35739026  656.35739026        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 8               return   -0.01465150  0.04162086   -0.15457815    0.17361128     3000
T Epoch 8         policy_ratio    0.99999377  0.00383151    0.98579383    1.01468182     3000
T Epoch 8          policy_loss   -0.00063859  0.00389436   -0.01358078    0.01609230     3000
T Epoch 8           value_loss    0.85469290  0.00961076    0.82341999    0.88652360     3000
T Epoch 8              entropy    1.67363449  0.02705224    1.58726835    1.74516618     3000
T Epoch 8                 loss    0.39323517  0.00632308    0.37307784    0.41430470     3000
T Epoch 8            grad_norm    0.63231367  0.18371966    0.19998580    1.67909014     3000
T Epoch 8  sample_data_time/ms    0.04502327  0.08096934    0.02422836    2.92344019     3000
T Epoch 8  batch_learn_time/ms   23.40464050  0.90434318   22.15095982   29.86885235     3000
T Epoch 8       episode_length    3.82031988  1.94164510    1.00000000   29.00000000   120670
T Epoch 8       episode_return   -0.02503754  1.00013370   -1.28000000    0.99000000   120670
T Epoch 8       episode_time/s    0.03064911  0.01597618    0.00601388    0.24140662   120670
T Epoch 8     steps_per_second  125.82796963  8.42141123   78.96359440  166.28211635   120670
T Epoch 8         correct_rate    0.50158283  0.49999749    0.00000000    1.00000000   120670
T Epoch 8                 time  730.29503339  0.00000000  730.29503339  730.29503339        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 8    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 8    episode_return    0.23000000  0.97077289   -1.01000000    0.99000000      100
E Epoch 8    episode_time/s    0.01586715  0.00087084    0.01459253    0.02005922      100
E Epoch 8  steps_per_second  126.39613685  6.39706361   99.70476708  137.05645850      100
E Epoch 8      correct_rate    0.62000000  0.48538644    0.00000000    1.00000000      100
E Epoch 8              time  733.30232875  0.00000000  733.30232875  733.30232875        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 9               return   -0.02087352  0.04117462   -0.15624823    0.10911963     3000
T Epoch 9         policy_ratio    0.99990458  0.00401915    0.98433357    1.01490498     3000
T Epoch 9          policy_loss   -0.00112256  0.00382774   -0.01475955    0.01295188     3000
T Epoch 9           value_loss    0.83958658  0.00972323    0.80585545    0.87268710     3000
T Epoch 9              entropy    1.71265100  0.01895551    1.63964951    1.76663756     3000
T Epoch 9                 loss    0.38441771  0.00618876    0.36352390    0.40436244     3000
T Epoch 9            grad_norm    0.63715222  0.17989188    0.26256919    1.48108649     3000
T Epoch 9  sample_data_time/ms    0.04163886  0.01586846    0.02394896    0.58636628     3000
T Epoch 9  batch_learn_time/ms   23.42588380  0.94273238   22.07701094   29.92305253     3000
T Epoch 9       episode_length    4.08631762  2.10423859    1.00000000   28.00000000   111333
T Epoch 9       episode_return   -0.03247097  1.00019552   -1.27000000    0.99000000   111333
T Epoch 9       episode_time/s    0.03285521  0.01733878    0.00558198    0.23115753   111333
T Epoch 9     steps_per_second  125.56050270  8.25881664   83.13413821  179.14794237   111333
T Epoch 9         correct_rate    0.49919611  0.49999935    0.00000000    1.00000000   111333
T Epoch 9                 time  807.32649377  0.00000000  807.32649377  807.32649377        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 9    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 9    episode_return   -0.21000000  0.97979590   -1.01000000    0.99000000      100
E Epoch 9    episode_time/s    0.01578222  0.00087567    0.01416495    0.01885246      100
E Epoch 9  steps_per_second  127.09951769  6.77255815  106.08697608  141.19362866      100
E Epoch 9      correct_rate    0.40000000  0.48989795    0.00000000    1.00000000      100
E Epoch 9              time  810.33389950  0.00000000  810.33389950  810.33389950        1


      info                  key          mean         std           min           max    count
----------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 10               return   -0.02263086  0.04164448   -0.16662507    0.13292439     3000
T Epoch 10         policy_ratio    0.99987087  0.00419425    0.98565757    1.01368642     3000
T Epoch 10          policy_loss   -0.00148837  0.00405800   -0.01715033    0.01348615     3000
T Epoch 10           value_loss    0.82781026  0.00931992    0.79588330    0.85999966     3000
T Epoch 10              entropy    1.73510259  0.01861863    1.67118227    1.79680717     3000
T Epoch 10                 loss    0.37771471  0.00621550    0.35566261    0.39957860     3000
T Epoch 10            grad_norm    0.63963440  0.17353505    0.26130050    1.49775851     3000
T Epoch 10  sample_data_time/ms    0.04348744  0.05558714    0.02363138    2.28964537     3000
T Epoch 10  batch_learn_time/ms   23.40955066  0.90271281   22.06121385   29.93332967     3000
T Epoch 10       episode_length    4.24656164  2.19738631    1.00000000   36.00000000   106955
T Epoch 10       episode_return   -0.03382133  1.00019423   -1.35000000    0.99000000   106955
T Epoch 10       episode_time/s    0.03419484  0.01812531    0.00665949    0.28651339   106955
T Epoch 10     steps_per_second  125.34990624  8.04470958   81.31889952  164.28838626   106955
T Epoch 10         correct_rate    0.49932214  0.49999954    0.00000000    1.00000000   106955
T Epoch 10                 time  884.32048856  0.00000000  884.32048856  884.32048856        1


      info               key          mean         std           min           max    count
----------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 10    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 10    episode_return   -0.07000000  0.99819838   -1.01000000    0.99000000      100
E Epoch 10    episode_time/s    0.01555542  0.00075645    0.01276384    0.01773884      100
E Epoch 10  steps_per_second  128.88569861  6.47211685  112.74696422  156.69267782      100
E Epoch 10      correct_rate    0.47000000  0.49909919    0.00000000    1.00000000      100
E Epoch 10              time  887.32673029  0.00000000  887.32673029  887.32673029        1


      info                  key          mean         std           min           max    count
----------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 11               return   -0.01482148  0.04019650   -0.13895199    0.12439085     3000
T Epoch 11         policy_ratio    0.99989590  0.00427188    0.98550284    1.01845574     3000
T Epoch 11          policy_loss   -0.00185954  0.00430307   -0.01729472    0.01550671     3000
T Epoch 11           value_loss    0.81209930  0.01417101    0.76350605    0.85612023     3000
T Epoch 11              entropy    1.75942378  0.02315854    1.66404796    1.81609499     3000
T Epoch 11                 loss    0.36900164  0.00857058    0.33971497    0.39528391     3000
T Epoch 11            grad_norm    0.63708289  0.16327504    0.30595350    1.59516144     3000
T Epoch 11  sample_data_time/ms    0.04398965  0.06637935    0.02426933    1.91031862     3000
T Epoch 11  batch_learn_time/ms   23.42774913  0.96044394   22.13526703   31.40726127     3000
T Epoch 11       episode_length    4.49557829  2.45525970    1.00000000   56.00000000   100866
T Epoch 11       episode_return   -0.03051425  0.99992978   -1.39000000    0.99000000   100866
T Epoch 11       episode_time/s    0.03629611  0.02026664    0.00635743    0.46971444   100866
T Epoch 11     steps_per_second  125.00166612  7.90222091   81.50016274  192.31831785   100866
T Epoch 11         correct_rate    0.50222077  0.49999507    0.00000000    1.00000000   100866
T Epoch 11                 time  961.35704870  0.00000000  961.35704870  961.35704870        1


      info               key          mean         std           min           max    count
----------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 11    episode_length    2.00000000  0.00000000    2.00000000    2.00000000      100
E Epoch 11    episode_return   -0.05000000  0.99919968   -1.01000000    0.99000000      100
E Epoch 11    episode_time/s    0.01558046  0.00069435    0.01341295    0.01732545      100
E Epoch 11  steps_per_second  128.62109204  5.73858634  115.43714366  149.10960219      100
E Epoch 11      correct_rate    0.48000000  0.49959984    0.00000000    1.00000000      100
E Epoch 11              time  964.36410927  0.00000000  964.36410927  964.36410927        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 12               return     0.02591952  0.04806420    -0.12658274     0.19599783     3000
T Epoch 12         policy_ratio     0.99985000  0.00484600     0.97844732     1.01593280     3000
T Epoch 12          policy_loss    -0.00302222  0.00475311    -0.02122399     0.01172394     3000
T Epoch 12           value_loss     0.75221914  0.03404593     0.63917804     0.82506269     3000
T Epoch 12              entropy     1.80372176  0.01658460     1.74014294     1.84782910     3000
T Epoch 12                 loss     0.33701292  0.01857991     0.27355894     0.37546819     3000
T Epoch 12            grad_norm     0.64827242  0.14658087     0.31204900     1.58029735     3000
T Epoch 12  sample_data_time/ms     0.04390272  0.08038037     0.02405141     2.85356026     3000
T Epoch 12  batch_learn_time/ms    23.34476832  0.85983968    22.01288659    28.59553508     3000
T Epoch 12       episode_length     5.36394924  3.38561119     1.00000000    64.00000000    84553
T Epoch 12       episode_return    -0.00911677  0.99751286    -2.63000000     0.99000000    84553
T Epoch 12       episode_time/s     0.04323728  0.02778565     0.00558668     0.54356467    84553
T Epoch 12     steps_per_second   125.27937410  7.63803695    82.34649645   178.99703617    84553
T Epoch 12         correct_rate     0.51728543  0.49970112     0.00000000     1.00000000    84551
T Epoch 12                 time  1038.18807858  0.00000000  1038.18807858  1038.18807858        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 12    episode_length     2.00000000  0.00000000     2.00000000     2.00000000      100
E Epoch 12    episode_return    -0.09000000  0.99679486    -1.01000000     0.99000000      100
E Epoch 12    episode_time/s     0.01539596  0.00074270     0.01359840     0.01781913      100
E Epoch 12  steps_per_second   130.19951109  6.13686880   112.23890265   147.07608449      100
E Epoch 12      correct_rate     0.46000000  0.49839743     0.00000000     1.00000000      100
E Epoch 12              time  1042.19872318  0.00000000  1042.19872318  1042.19872318        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 13               return     0.45762576  0.21635688     0.04727705     0.86512357     3000
T Epoch 13         policy_ratio     0.99932142  0.00727147     0.97312808     1.02588272     3000
T Epoch 13          policy_loss    -0.01208286  0.00718210    -0.03482125     0.01101097     3000
T Epoch 13           value_loss     0.41484202  0.16957075     0.06553015     0.71165693     3000
T Epoch 13              entropy     1.62044774  0.14129248     1.29534078     1.82317674     3000
T Epoch 13                 loss     0.16292920  0.08527402    -0.00369635     0.31788650     3000
T Epoch 13            grad_norm     0.63281636  0.11155525     0.36850819     1.29135382     3000
T Epoch 13  sample_data_time/ms     0.04518969  0.08460551     0.02439972     2.43621878     3000
T Epoch 13  batch_learn_time/ms    23.31023528  0.85970164    21.99824248    29.81940843     3000
T Epoch 13       episode_length     8.37490212  4.24953470     1.00000000    64.00000000    54913
T Epoch 13       episode_return     0.38495839  0.87515136    -2.63000000     0.99000000    54913
T Epoch 13       episode_time/s     0.06749715  0.03461214     0.00524506     0.52846057    54913
T Epoch 13     steps_per_second   124.85721705  6.22995163    83.58073281   190.65566588    54913
T Epoch 13         correct_rate     0.72939848  0.44427057     0.00000000     1.00000000    54911
T Epoch 13                 time  1115.87734411  0.00000000  1115.87734411  1115.87734411        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 13    episode_length     7.37000000  0.48280431     7.00000000     8.00000000      100
E Epoch 13    episode_return     0.93630000  0.00482804     0.93000000     0.94000000      100
E Epoch 13    episode_time/s     0.05862165  0.00422568     0.05111074     0.06647185      100
E Epoch 13  steps_per_second   125.82616849  3.46872438   120.35169684   136.95751915      100
E Epoch 13      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 13              time  1120.88632742  0.00000000  1120.88632742  1120.88632742        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 14               return     0.92732239  0.03130451     0.79637438     0.96793318     3000
T Epoch 14         policy_ratio     0.99996094  0.00786797     0.97287565     1.02416539     3000
T Epoch 14          policy_loss    -0.00847124  0.00747973    -0.03274520     0.03390277     3000
T Epoch 14           value_loss     0.02513873  0.02490570     0.00075494     0.13242728     3000
T Epoch 14              entropy     0.98181588  0.17477277     0.64980519     1.32555699     3000
T Epoch 14                 loss    -0.01553820  0.01071203    -0.04057550     0.03181831     3000
T Epoch 14            grad_norm     0.49144409  0.20005501     0.09558558     3.03933668     3000
T Epoch 14  sample_data_time/ms     0.04127226  0.01602822     0.02446398     0.51063392     3000
T Epoch 14  batch_learn_time/ms    23.33927073  0.89351371    21.98893763    29.20453623     3000
T Epoch 14       episode_length     8.43436174  1.56114906     1.00000000    27.00000000    55166
T Epoch 14       episode_return     0.90752928  0.18815438    -1.26000000     0.99000000    55166
T Epoch 14       episode_time/s     0.06819283  0.01285651     0.00538476     0.21722290    55166
T Epoch 14     steps_per_second   123.87267214  4.65645309    96.19579781   185.70917039    55166
T Epoch 14         correct_rate     0.99093645  0.09477028     0.00000000     1.00000000    55166
T Epoch 14                 time  1194.69934776  0.00000000  1194.69934776  1194.69934776        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 14    episode_length     7.00000000  0.00000000     7.00000000     7.00000000      100
E Epoch 14    episode_return     0.94000000  0.00000000     0.94000000     0.94000000      100
E Epoch 14    episode_time/s     0.05617841  0.00108617     0.05254618     0.05857052      100
E Epoch 14  steps_per_second   124.65034510  2.44894380   119.51405471   133.21615660      100
E Epoch 14      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 14              time  1198.70790191  0.00000000  1198.70790191  1198.70790191        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 15               return     0.97045428  0.00487579     0.95014745     0.98216671     3000
T Epoch 15         policy_ratio     0.99921874  0.00590893     0.97703964     1.02243257     3000
T Epoch 15          policy_loss    -0.00109507  0.00937818    -0.01774572     0.17033401     3000
T Epoch 15           value_loss     0.00369792  0.00410301     0.00028799     0.02802260     3000
T Epoch 15              entropy     0.54544211  0.06075222     0.41537365     0.72054583     3000
T Epoch 15                 loss    -0.01015495  0.00949992    -0.02840818     0.16094835     3000
T Epoch 15            grad_norm     0.39151693  0.33383564     0.05890456     9.63900375     3000
T Epoch 15  sample_data_time/ms     0.04157247  0.03458424     0.02285372     1.82024017     3000
T Epoch 15  batch_learn_time/ms    23.37566614  0.88903505    21.77562006    32.28627238     3000
T Epoch 15       episode_length     7.19347367  0.58854493     1.00000000    64.00000000    63895
T Epoch 15       episode_return     0.93573331  0.06860706    -2.63000000     0.98000000    63895
T Epoch 15       episode_time/s     0.05814357  0.00534975     0.00484634     0.52937487    63895
T Epoch 15     steps_per_second   123.92125506  4.85260323    99.05941919   206.34119899    63895
T Epoch 15         correct_rate     0.99885748  0.03378183     0.00000000     1.00000000    63894
T Epoch 15                 time  1272.58110178  0.00000000  1272.58110178  1272.58110178        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 15    episode_length     7.00000000  0.00000000     7.00000000     7.00000000      100
E Epoch 15    episode_return     0.94000000  0.00000000     0.94000000     0.94000000      100
E Epoch 15    episode_time/s     0.05589907  0.00104306     0.05154415     0.05850973      100
E Epoch 15  steps_per_second   125.26937867  2.34278761   119.63822888   135.80589304      100
E Epoch 15      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 15              time  1276.59244791  0.00000000  1276.59244791  1276.59244791        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 16               return     0.98131113  0.00363443     0.96434659     0.99127817     3000
T Epoch 16         policy_ratio     0.99994176  0.01414027     0.97841299     1.67920697     3000
T Epoch 16          policy_loss     0.00166026  0.02584591    -0.01535409     1.21703637     3000
T Epoch 16           value_loss     0.00240760  0.00341780     0.00014256     0.02371368     3000
T Epoch 16              entropy     0.45305476  0.02460257     0.36397403     0.53932905     3000
T Epoch 16                 loss    -0.00619704  0.02588834    -0.02433160     1.21222389     3000
T Epoch 16            grad_norm     0.37760111  1.58033129     0.04610204    79.09159088     3000
T Epoch 16  sample_data_time/ms     0.04425403  0.07718048     0.02360251     2.49823742     3000
T Epoch 16  batch_learn_time/ms    23.27753880  0.82513486    21.90615796    29.87651434     3000
T Epoch 16       episode_length     7.03029418  0.30828469     2.00000000    47.00000000    65029
T Epoch 16       episode_return     0.93800551  0.05823916    -1.46000000     0.98000000    65029
T Epoch 16       episode_time/s     0.05689297  0.00341309     0.01307951     0.38789110    65029
T Epoch 16     steps_per_second   123.76660284  4.84924649    97.37110168   152.91094804    65029
T Epoch 16         correct_rate     0.99915422  0.02906994     0.00000000     1.00000000    65029
T Epoch 16                 time  1350.15494648  0.00000000  1350.15494648  1350.15494648        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 16    episode_length     7.00000000  0.00000000     7.00000000     7.00000000      100
E Epoch 16    episode_return     0.94000000  0.00000000     0.94000000     0.94000000      100
E Epoch 16    episode_time/s     0.05591220  0.00076626     0.05338570     0.05740127      100
E Epoch 16  steps_per_second   125.22009388  1.73765009   121.94851413   131.12125279      100
E Epoch 16      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 16              time  1354.16277246  0.00000000  1354.16277246  1354.16277246        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 17               return     0.99085049  0.00373198     0.97650695     1.00084877     3000
T Epoch 17         policy_ratio     0.99948037  0.00835138     0.97165066     1.17484164     3000
T Epoch 17          policy_loss     0.00061178  0.01067294    -0.01303740     0.29032999     3000
T Epoch 17           value_loss     0.00310756  0.00498246     0.00008382     0.03856286     3000
T Epoch 17              entropy     0.43124545  0.02429732     0.34880772     0.50746167     3000
T Epoch 17                 loss    -0.00645935  0.01059776    -0.01999223     0.28158736     3000
T Epoch 17            grad_norm     0.31164021  0.71917331     0.04051616    31.83717346     3000
T Epoch 17  sample_data_time/ms     0.04169877  0.04647750     0.02285186     1.87334698     3000
T Epoch 17  batch_learn_time/ms    23.28027090  0.80881654    21.94199618    29.87246495     3000
T Epoch 17       episode_length     7.01177192  0.19127822     1.00000000    39.00000000    65240
T Epoch 17       episode_return     0.93736849  0.07053572    -1.08000000     0.96000000    65240
T Epoch 17       episode_time/s     0.05676282  0.00278552     0.01179049     0.31704194    65240
T Epoch 17     steps_per_second   123.72885185  4.95649252    84.81413884   148.15758503    65240
T Epoch 17         correct_rate     0.99874310  0.03543046     0.00000000     1.00000000    65240
T Epoch 17                 time  1427.80443594  0.00000000  1427.80443594  1427.80443594        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 17    episode_length     7.00000000  0.00000000     7.00000000     7.00000000      100
E Epoch 17    episode_return     0.94000000  0.00000000     0.94000000     0.94000000      100
E Epoch 17    episode_time/s     0.05626830  0.00110967     0.05333528     0.05918291      100
E Epoch 17  steps_per_second   124.45275458  2.47457992   118.27738363   131.24519770      100
E Epoch 17      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 17              time  1432.81374954  0.00000000  1432.81374954  1432.81374954        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 18               return     1.00267653  0.00410401     0.98477805     1.01311851     3000
T Epoch 18         policy_ratio     0.99954000  0.00577825     0.98350823     1.09693122     3000
T Epoch 18          policy_loss     0.00083596  0.00872672    -0.01264398     0.24810158     3000
T Epoch 18           value_loss     0.00120575  0.00284168     0.00005056     0.02281380     3000
T Epoch 18              entropy     0.42364919  0.02451985     0.34303483     0.50611919     3000
T Epoch 18                 loss    -0.00703415  0.00869169    -0.01953890     0.23908126     3000
T Epoch 18            grad_norm     0.27074914  0.37762813     0.03197153    13.07730961     3000
T Epoch 18  sample_data_time/ms     0.04271509  0.05904501     0.02395734     1.69106200     3000
T Epoch 18  batch_learn_time/ms    23.32920497  0.85335533    22.03762159    27.83215605     3000
T Epoch 18       episode_length     7.00504913  0.10376973     2.00000000    17.00000000    66150
T Epoch 18       episode_return     0.93898201  0.04387208    -1.16000000     0.96000000    66150
T Epoch 18       episode_time/s     0.05676765  0.00243705     0.01454270     0.14401022    66150
T Epoch 18     steps_per_second   123.59484666  4.87957528    99.14557945   143.39691810    66150
T Epoch 18         correct_rate     0.99951625  0.02198898     0.00000000     1.00000000    66150
T Epoch 18                 time  1506.53678989  0.00000000  1506.53678989  1506.53678989        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 18    episode_length     7.00000000  0.00000000     7.00000000     7.00000000      100
E Epoch 18    episode_return     0.94000000  0.00000000     0.94000000     0.94000000      100
E Epoch 18    episode_time/s     0.05626101  0.00127536     0.05271818     0.05848889      100
E Epoch 18  steps_per_second   124.48460288  2.84674562   119.68085724   132.78150867      100
E Epoch 18      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 18              time  1510.54568011  0.00000000  1510.54568011  1510.54568011        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 19               return     1.01259579  0.00444234     0.99183458     1.02473855     3000
T Epoch 19         policy_ratio     0.99853079  0.00771239     0.95562530     1.13800120     3000
T Epoch 19          policy_loss     0.00052690  0.02780103    -0.01705920     1.43655169     3000
T Epoch 19           value_loss     0.00384218  0.00589971     0.00005058     0.04034331     3000
T Epoch 19              entropy     0.40691333  0.02718904     0.32482302     0.55655873     3000
T Epoch 19                 loss    -0.00569028  0.02784235    -0.02309475     1.43532157     3000
T Epoch 19            grad_norm     0.33512056  4.23730155     0.01554179   231.34500122     3000
T Epoch 19  sample_data_time/ms     0.04308336  0.09525782     0.02319179     3.95705365     3000
T Epoch 19  batch_learn_time/ms    23.30089972  0.83716893    21.88708819    30.24135157     3000
T Epoch 19       episode_length     7.01601795  0.35890027     3.00000000    64.00000000    65052
T Epoch 19       episode_return     0.93739562  0.07070615    -2.63000000     0.95000000    65052
T Epoch 19       episode_time/s     0.05692021  0.00375177     0.02178258     0.52586697    65052
T Epoch 19     steps_per_second   123.46124773  4.91475355   101.35146293   143.93907496    65052
T Epoch 19         correct_rate     0.99880094  0.03460667     0.00000000     1.00000000    65051
T Epoch 19                 time  1584.17634783  0.00000000  1584.17634783  1584.17634783        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 19    episode_length     7.00000000  0.00000000     7.00000000     7.00000000      100
E Epoch 19    episode_return     0.94000000  0.00000000     0.94000000     0.94000000      100
E Epoch 19    episode_time/s     0.05660744  0.00126155     0.05339714     0.06063637      100
E Epoch 19  steps_per_second   123.71998894  2.75344552   115.44227477   131.09316913      100
E Epoch 19      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 19              time  1588.18417420  0.00000000  1588.18417420  1588.18417420        1

```
