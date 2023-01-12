# Table VII: comparison of PLRU with and without PLCache

We compare the attack patterns found in Table VII and epochs need for PL cache and normal PLRU cache.


![](../../fig/table7.png)

First, go to the directory.

```
cd ${GIT_ROOT}/src/rlmeta
```

(Optional) To train a config in Table VII, use the following script:

```
$ python train_ppo_attack.py env_config=<NAME_OF_THE_CONFIG>
```

There are 2 configs in Table VII, and we have ```hpca_ae_exp_7_baseline```, ```hpca_ae_exp_7_pl``` correpondingly, replace ```<NAME_OF_THE_CONFIG>``` with these.

Use ```Ctrl+C``` to interrupt the training, which will save a checkpoint in the given path.

Since the training takes some time, we provide pretrained checkpoints in the following directory ```src/rlmeta/data/table7```. For each folder, we include the ```ppo_agent-X.pth``` which is the last checkpoint in the training processes, we also include the training logs.


Use the following to grep the interested information from ```train_ppo_attack.log```

```
$ cat train_ppo_attack.log |grep Eval | awk '{print  $4 $5 $6 $7 $8 $9 $10 $11 $12 $13 $14 $15 $16 $56 $57 $58 $59 $60 $61 $62 $63 $64 $65 $66 $67 $68 $69 $70 $71 $72} '
```

which will print out the mean/max/min of ```episode_length```, ```correct_rate``` for the each epoch.

For example, run the following script will print the logs shown below.

```
$ python data/show_log.py --log_file=data/table7/hpca_ae_exp_7_baseline/exp_1/train_ppo_attack.log
```

You can see that at epoch **6** which is the **7th** epoch, the training ```correct_rate``` jump to 0.98736399 which is over 0.97. And at th end of the training, the evaluation ```episode_length``` is 6.98. For baseline and PLcache, we perform three experiments each and calculate the corresponding average epoch to converge and final evaluation episode_length at the end.

```
Experiment Configs = {'env_config': {'length_violation_reward': -2.0, 'double_victim_access_reward': -0.01, 'victim_access_reward': -0.01, 'correct_reward': 1.0, 'wrong_reward': -1.0, 'step_reward': -0.01, 'verbose': 0, 'force_victim_hit': False, 'flush_inst': False, 'allow_victim_multi_access': True, 'allow_empty_victim_access': True, 'attacker_addr_s': 1, 'attacker_addr_e': 5, 'victim_addr_s': 0, 'victim_addr_e': 0, 'reset_limit': 1, 'cache_configs': {'architecture': {'word_size': 1, 'block_size': 1, 'write_back': True}, 'cache_1': {'blocks': 4, 'associativity': 4, 'hit_time': 1, 'rep_policy': 'tree-plru', 'prefetcher': 'none'}, 'mem': {'hit_time': 1000}}, 'window_size': 64}, 'model_config': {'type': 'transformer', 'args': {'latency_dim': 3, 'victim_acc_dim': 2, 'action_dim': 64, 'step_dim': 64, 'action_embed_dim': 16, 'step_embed_dim': 4, 'hidden_dim': 128, 'output_dim': -1, 'num_layers': 1}}, 'm_server_name': 'm_server', 'm_server_addr': '127.0.0.1:4411', 'r_server_name': 'r_server', 'r_server_addr': '127.0.0.1:4412', 'c_server_name': 'c_server', 'c_server_addr': '127.0.0.1:4413', 'train_device': 'cuda:0', 'infer_device': 'cuda:1', 'num_train_rollouts': 48, 'num_train_workers': 24, 'num_eval_rollouts': 4, 'num_eval_workers': 2, 'replay_buffer_size': 131072, 'prefetch': 2, 'batch_size': 512, 'optimizer': {'name': 'Adam', 'lr': 0.0001}, 'learning_starts': 65536, 'model_push_period': 10, 'entropy_coeff': 0.02, 'num_epochs': 1000, 'steps_per_epoch': 3000, 'num_eval_episodes': 100, 'seed': 111, 'table_view': False}

     info                  key          mean          std          min           max    count
---------  -------------------  ------------  -----------  -----------  ------------  -------
T Epoch 0               return   -0.19987801   0.19155996  -1.54232836    0.03486374     3000
T Epoch 0         policy_ratio    1.00012223   0.00689717   0.97221738    1.02276063     3000
T Epoch 0          policy_loss   -0.00475277   0.00668502  -0.02772057    0.03881596     3000
T Epoch 0           value_loss    3.16264567   8.35696581   0.62967324   94.91001892     3000
T Epoch 0              entropy    1.91094450   0.03295782   1.84328508    2.00610876     3000
T Epoch 0                 loss    1.53835118   4.17954679   0.26851571   47.41218185     3000
T Epoch 0            grad_norm    1.19655674   1.67401986   0.25490147   22.17344856     3000
T Epoch 0  sample_data_time/ms    0.04831568   0.18501433   0.02571335    9.79752373     3000
T Epoch 0  batch_learn_time/ms   23.80603842  13.49408063  22.15815615  760.27040090     3000
T Epoch 0       episode_length    7.53744747   5.21863314   1.00000000   54.00000000    67815
T Epoch 0       episode_return   -0.28943995   0.96632760  -1.40000000    0.99000000    67815
T Epoch 0       episode_time/s    0.06023575   0.05218127   0.00425532    2.08762649    67815
T Epoch 0     steps_per_second  127.88911410  10.80491542   0.50351821  235.00000744    67815
T Epoch 0         correct_rate    0.38796726   0.48728705   0.00000000    1.00000000    67815
T Epoch 0                 time   97.20001534   0.00000000  97.20001534   97.20001534        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 0    episode_length   64.00000000  0.00000000   64.00000000   64.00000000      100
E Epoch 0    episode_return   -2.63000000  0.00000000   -2.63000000   -2.63000000      100
E Epoch 0    episode_time/s    0.48979679  0.00587037    0.47632000    0.50278650      100
E Epoch 0  steps_per_second  130.68532065  1.57592838  127.29060993  134.36345199      100
E Epoch 0              time  113.22372672  0.00000000  113.22372672  113.22372672        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 1               return   -0.03633363  0.03897914   -0.16039342    0.09482402     3000
T Epoch 1         policy_ratio    0.99995027  0.00461934    0.98428893    1.01445603     3000
T Epoch 1          policy_loss   -0.00043032  0.00452858   -0.02086654    0.01595735     3000
T Epoch 1           value_loss    0.72503950  0.01450958    0.67706120    0.77447343     3000
T Epoch 1              entropy    1.87523260  0.01446489    1.82286477    1.91995525     3000
T Epoch 1                 loss    0.32458478  0.00868017    0.29266614    0.35152477     3000
T Epoch 1            grad_norm    0.72135961  0.23212697    0.24452177    1.85427058     3000
T Epoch 1  sample_data_time/ms    0.04236950  0.01436627    0.02515223    0.49807131     3000
T Epoch 1  batch_learn_time/ms   23.43600527  0.95389554   22.13705005   29.44972599     3000
T Epoch 1       episode_length    6.93287992  4.23818084    1.00000000   48.00000000    78516
T Epoch 1       episode_return   -0.05637399  1.00005804   -1.41000000    0.99000000    78516
T Epoch 1       episode_time/s    0.05460595  0.03360181    0.00478154    0.37668908    78516
T Epoch 1     steps_per_second  127.66510448  7.56745685   72.82295222  209.13782439    78516
T Epoch 1         correct_rate    0.50147741  0.49999782    0.00000000    1.00000000    78516
T Epoch 1                 time  187.19797604  0.00000000  187.19797604  187.19797604        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 1    episode_length    7.12000000  1.99639675    5.00000000    9.00000000      100
E Epoch 1    episode_return   -0.12120000  0.99912690   -1.08000000    0.96000000      100
E Epoch 1    episode_time/s    0.05484773  0.01569742    0.03468202    0.07361033      100
E Epoch 1  steps_per_second  130.16099025  5.15708287  117.75441805  144.80659567      100
E Epoch 1      correct_rate    0.47000000  0.49909919    0.00000000    1.00000000      100
E Epoch 1              time  191.20665655  0.00000000  191.20665655  191.20665655        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 2               return    0.02227490  0.04286617   -0.13552272    0.17346478     3000
T Epoch 2         policy_ratio    0.99993740  0.00462334    0.98321211    1.01715076     3000
T Epoch 2          policy_loss   -0.00044541  0.00454102   -0.01548125    0.01682519     3000
T Epoch 2           value_loss    0.69034131  0.02601845    0.61214381    0.75608391     3000
T Epoch 2              entropy    1.89239623  0.01224678    1.84577405    1.92739618     3000
T Epoch 2                 loss    0.30687732  0.01399711    0.26326188    0.34866703     3000
T Epoch 2            grad_norm    0.68649241  0.19079493    0.28460059    1.64742351     3000
T Epoch 2  sample_data_time/ms    0.04481737  0.09380738    0.02484303    3.50885419     3000
T Epoch 2  batch_learn_time/ms   23.37804453  0.94446784   22.15531003   29.65128701     3000
T Epoch 2       episode_length    7.46722702  4.89205989    1.00000000   64.00000000    63055
T Epoch 2       episode_return   -0.02637237  0.99698155   -2.63000000    0.99000000    63055
T Epoch 2       episode_time/s    0.05877972  0.03876913    0.00501162    0.52193048    63055
T Epoch 2     steps_per_second  127.71634310  7.34051453   87.43115906  199.53638893    63055
T Epoch 2         correct_rate    0.51918228  0.49963190    0.00000000    1.00000000    63053
T Epoch 2                 time  264.98863540  0.00000000  264.98863540  264.98863540        1


     info               key          mean          std           min           max    count
---------  ----------------  ------------  -----------  ------------  ------------  -------
E Epoch 2    episode_length    9.80000000  14.06484980    4.00000000   54.00000000      100
E Epoch 2    episode_return   -0.22800000   1.02134323   -1.53000000    0.97000000      100
E Epoch 2    episode_time/s    0.07427685   0.10693866    0.02833031    0.41684624      100
E Epoch 2  steps_per_second  132.37370876   3.47404555  123.34843009  141.19152604      100
E Epoch 2      correct_rate    0.43000000   0.49507575    0.00000000    1.00000000      100
E Epoch 2              time  269.99695429   0.00000000  269.99695429  269.99695429        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 3               return    0.13580148  0.05711672   -0.03372654    0.29705763     3000
T Epoch 3         policy_ratio    0.99987226  0.00507048    0.98080170    1.01627147     3000
T Epoch 3          policy_loss   -0.00141670  0.00510325   -0.01788009    0.01797792     3000
T Epoch 3           value_loss    0.56264221  0.04965419    0.43024123    0.67922437     3000
T Epoch 3              entropy    1.88641011  0.01443979    1.84718347    1.92651808     3000
T Epoch 3                 loss    0.24217620  0.02550459    0.17101195    0.30870652     3000
T Epoch 3            grad_norm    0.71717566  0.16968805    0.34349865    1.51554930     3000
T Epoch 3  sample_data_time/ms    0.04424427  0.06459697    0.02330216    2.05591973     3000
T Epoch 3  batch_learn_time/ms   23.40406530  0.98091860   21.81584202   32.52789378     3000
T Epoch 3       episode_length   10.46727911  7.05143482    1.00000000   64.00000000    45827
T Epoch 3       episode_return    0.05194383  0.98259428   -2.63000000    0.99000000    45827
T Epoch 3       episode_time/s    0.08211484  0.05539274    0.00544904    0.51701255    45827
T Epoch 3     steps_per_second  127.77928705  6.16774184   85.93017939  183.51869486    45827
T Epoch 3         correct_rate    0.57357692  0.49455681    0.00000000    1.00000000    45816
T Epoch 3                 time  343.84449909  0.00000000  343.84449909  343.84449909        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 3    episode_length   14.83000000  7.51139801    7.00000000   30.00000000      100
E Epoch 3    episode_return    0.72170000  0.49408310   -1.06000000    0.94000000      100
E Epoch 3    episode_time/s    0.11024645  0.05601899    0.05050123    0.22853589      100
E Epoch 3  steps_per_second  134.62328908  2.26727581  128.28880058  139.88408738      100
E Epoch 3      correct_rate    0.93000000  0.25514702    0.00000000    1.00000000      100
E Epoch 3              time  349.85373930  0.00000000  349.85373930  349.85373930        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 4               return    0.35854978  0.07912445    0.15783684    0.56057131     3000
T Epoch 4         policy_ratio    1.00007300  0.00586152    0.98180580    1.02317643     3000
T Epoch 4          policy_loss   -0.00318336  0.00582066   -0.02531348    0.01926736     3000
T Epoch 4           value_loss    0.38628328  0.05066017    0.23613872    0.52151525     3000
T Epoch 4              entropy    1.77671754  0.04743664    1.66705847    1.86474717     3000
T Epoch 4                 loss    0.15442393  0.02530753    0.07140230    0.21818095     3000
T Epoch 4            grad_norm    0.83791860  0.21983628    0.43464041    3.12579942     3000
T Epoch 4  sample_data_time/ms    0.04378206  0.05799809    0.02515782    2.17402214     3000
T Epoch 4  batch_learn_time/ms   23.36884734  0.89668498   22.10400067   29.63515371     3000
T Epoch 4       episode_length   15.21156254  7.78418652    1.00000000   64.00000000    31948
T Epoch 4       episode_return    0.28858426  0.89452860   -2.63000000    0.99000000    31948
T Epoch 4       episode_time/s    0.11929739  0.06099356    0.00579684    0.50448375    31948
T Epoch 4     steps_per_second  127.57495573  4.58761313   85.54373605  172.50778163    31948
T Epoch 4         correct_rate    0.71553343  0.45115999    0.00000000    1.00000000    31944
T Epoch 4                 time  423.57101217  0.00000000  423.57101217  423.57101217        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 4    episode_length   12.65000000  3.57875677    8.00000000   20.00000000      100
E Epoch 4    episode_return    0.88350000  0.03578757    0.81000000    0.93000000      100
E Epoch 4    episode_time/s    0.09443435  0.02682562    0.05712114    0.15048390      100
E Epoch 4  steps_per_second  134.03554128  2.28002395  127.56926099  140.05322963      100
E Epoch 4      correct_rate    1.00000000  0.00000000    1.00000000    1.00000000      100
E Epoch 4              time  428.57955731  0.00000000  428.57955731  428.57955731        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 5               return    0.73730704  0.10685432    0.47151160    0.91069663     3000
T Epoch 5         policy_ratio    0.99987519  0.00817972    0.96789765    1.03983879     3000
T Epoch 5          policy_loss   -0.00365230  0.00830490   -0.02611106    0.08007199     3000
T Epoch 5           value_loss    0.15282870  0.06938502    0.01738167    0.35867807     3000
T Epoch 5              entropy    1.50611645  0.09380505    1.30902219    1.68782139     3000
T Epoch 5                 loss    0.04263972  0.03321934   -0.03001642    0.16509354     3000
T Epoch 5            grad_norm    0.97454300  0.52880173    0.41683528   12.69456100     3000
T Epoch 5  sample_data_time/ms    0.04310493  0.04915011    0.02483604    1.76102715     3000
T Epoch 5  batch_learn_time/ms   23.36150537  0.89867520   21.95623191   29.59366003     3000
T Epoch 5       episode_length   15.27855207  5.35087814    1.00000000   64.00000000    31355
T Epoch 5       episode_return    0.70986956  0.52362778   -2.63000000    0.99000000    31355
T Epoch 5       episode_time/s    0.12008426  0.04191596    0.00610731    0.50906476    31355
T Epoch 5     steps_per_second  127.27630553  3.95324569   96.54994404  163.73826321    31355
T Epoch 5         correct_rate    0.92638898  0.26113682    0.00000000    1.00000000    31354
T Epoch 5                 time  502.33115365  0.00000000  502.33115365  502.33115365        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 5    episode_length   11.45000000  1.71099386    9.00000000   15.00000000      100
E Epoch 5    episode_return    0.89550000  0.01710994    0.86000000    0.92000000      100
E Epoch 5    episode_time/s    0.08533450  0.01291228    0.06377809    0.11627202      100
E Epoch 5  steps_per_second  134.24441125  2.60230542  126.00530504  141.11429008      100
E Epoch 5      correct_rate    1.00000000  0.00000000    1.00000000    1.00000000      100
E Epoch 5              time  507.33981186  0.00000000  507.33981186  507.33981186        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 6               return    0.95279077  0.03279456    0.86148632    1.01732492     3000
T Epoch 6         policy_ratio    0.99941958  0.01053173    0.96343333    1.06544018     3000
T Epoch 6          policy_loss   -0.00071091  0.01486983   -0.02853899    0.41287619     3000
T Epoch 6           value_loss    0.03805198  0.01738169    0.00224842    0.10912268     3000
T Epoch 6              entropy    1.15907067  0.10080492    0.94669843    1.36620235     3000
T Epoch 6                 loss   -0.00486633  0.01669852   -0.03252233    0.41571465     3000
T Epoch 6            grad_norm    1.02806429  1.27969804    0.22180665   35.41819382     3000
T Epoch 6  sample_data_time/ms    0.04218688  0.01280581    0.02501532    0.21849293     3000
T Epoch 6  batch_learn_time/ms   23.34170921  0.83343800   22.21650677   28.95706426     3000
T Epoch 6       episode_length   11.91884574  2.72271000    1.00000000   64.00000000    39887
T Epoch 6       episode_return    0.86546494  0.22584886   -2.63000000    0.99000000    39887
T Epoch 6       episode_time/s    0.09415583  0.02134746    0.00658166    0.50029760    39887
T Epoch 6     steps_per_second  126.63876347  4.08880590  107.60167655  151.93728038    39887
T Epoch 6         correct_rate    0.98736399  0.11169756    0.00000000    1.00000000    39886
T Epoch 6                 time  580.96951759  0.00000000  580.96951759  580.96951759        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 6    episode_length    9.17000000  0.60091597    8.00000000   10.00000000      100
E Epoch 6    episode_return    0.91830000  0.00600916    0.91000000    0.93000000      100
E Epoch 6    episode_time/s    0.06862116  0.00466054    0.05636302    0.07789699      100
E Epoch 6  steps_per_second  133.70272758  3.51141488  122.86694904  141.93703579      100
E Epoch 6      correct_rate    1.00000000  0.00000000    1.00000000    1.00000000      100
E Epoch 6              time  585.97945136  0.00000000  585.97945136  585.97945136        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 7               return    1.03969302  0.01948340    0.97718811    1.07877982     3000
T Epoch 7         policy_ratio    0.99929957  0.01111676    0.96571279    1.07555401     3000
T Epoch 7          policy_loss   -0.00063547  0.01316941   -0.02052718    0.36626405     3000
T Epoch 7           value_loss    0.01600389  0.01053450    0.00058768    0.07136883     3000
T Epoch 7              entropy    0.76211015  0.11933915    0.51099086    1.02825344     3000
T Epoch 7                 loss   -0.00787573  0.01393264   -0.03072881    0.35835776     3000
T Epoch 7            grad_norm    0.92037749  1.58394912    0.10915446   46.24579239     3000
T Epoch 7  sample_data_time/ms    0.04452993  0.06561361    0.02418319    1.99961010     3000
T Epoch 7  batch_learn_time/ms   23.41141066  0.93195390   22.23758167   29.28967914     3000
T Epoch 7       episode_length    9.44185215  1.33773586    1.00000000   46.00000000    50363
T Epoch 7       episode_return    0.90624923  0.13672507   -1.23000000    0.96000000    50363
T Epoch 7       episode_time/s    0.07472601  0.01076994    0.00551651    0.36035513    50363
T Epoch 7     steps_per_second  126.48484990  4.47801524  105.63486864  181.27406103    50363
T Epoch 7         correct_rate    0.99533388  0.06814948    0.00000000    1.00000000    50363
T Epoch 7                 time  659.87279103  0.00000000  659.87279103  659.87279103        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 7    episode_length    8.09000000  0.34914181    7.00000000    9.00000000      100
E Epoch 7    episode_return    0.92910000  0.00349142    0.92000000    0.94000000      100
E Epoch 7    episode_time/s    0.06086704  0.00311676    0.05129309    0.06914440      100
E Epoch 7  steps_per_second  133.03265244  4.19039828  121.47618664  142.17469709      100
E Epoch 7      correct_rate    1.00000000  0.00000000    1.00000000    1.00000000      100
E Epoch 7              time  663.87968124  0.00000000  663.87968124  663.87968124        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 8               return    1.09896190  0.01636238    1.05670333    1.13481736     3000
T Epoch 8         policy_ratio    0.99834849  0.01422496    0.96308398    1.20743060     3000
T Epoch 8          policy_loss    0.00028876  0.01776582   -0.01862288    0.65361208     3000
T Epoch 8           value_loss    0.01113947  0.00916193    0.00033088    0.05736126     3000
T Epoch 8              entropy    0.39888072  0.09377664    0.18788400    0.58923894     3000
T Epoch 8                 loss   -0.00211912  0.01805194   -0.02563701    0.66104215     3000
T Epoch 8            grad_norm    0.71956425  1.40547538    0.07127929   33.97722244     3000
T Epoch 8  sample_data_time/ms    0.04325473  0.04530249    0.02482720    1.82022993     3000
T Epoch 8  batch_learn_time/ms   23.37752426  0.89218363   21.95918001   29.82483618     3000
T Epoch 8       episode_length    7.74976100  0.84567245    1.00000000   20.00000000    60670
T Epoch 8       episode_return    0.92660162  0.10846321   -1.16000000    0.96000000    60670
T Epoch 8       episode_time/s    0.06116132  0.00704967    0.00875416    0.15415528    60670
T Epoch 8     steps_per_second  126.91068719  5.26230178  105.69249589  147.24116957    60670
T Epoch 8         correct_rate    0.99704961  0.05423728    0.00000000    1.00000000    60670
T Epoch 8                 time  737.71268142  0.00000000  737.71268142  737.71268142        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 8    episode_length    6.97000000  0.17058722    6.00000000    7.00000000      100
E Epoch 8    episode_return    0.94030000  0.00170587    0.94000000    0.95000000      100
E Epoch 8    episode_time/s    0.05368235  0.00225771    0.04531696    0.05756798      100
E Epoch 8  steps_per_second  129.99714926  4.69763092  121.59537072  142.61691568      100
E Epoch 8      correct_rate    1.00000000  0.00000000    1.00000000    1.00000000      100
E Epoch 8              time  741.72052525  0.00000000  741.72052525  741.72052525        1


     info                  key          mean         std           min           max    count
---------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 9               return    1.14971109  0.01199847    1.11764741    1.17524862     3000
T Epoch 9         policy_ratio    0.99982983  0.00628881    0.97692949    1.12555540     3000
T Epoch 9          policy_loss    0.00029862  0.01385582   -0.01818046    0.45769835     3000
T Epoch 9           value_loss    0.00563256  0.00702464    0.00010794    0.04247653     3000
T Epoch 9              entropy    0.12430907  0.03244409    0.06431455    0.24508356     3000
T Epoch 9                 loss    0.00062872  0.01418838   -0.02204187    0.46103343     3000
T Epoch 9            grad_norm    0.40260532  1.64786385    0.01990981   65.33003235     3000
T Epoch 9  sample_data_time/ms    0.04358079  0.05599504    0.02307305    2.35431595     3000
T Epoch 9  batch_learn_time/ms   23.41496936  0.92449382   22.22724911   30.74162081     3000
T Epoch 9       episode_length    6.96071111  0.33439400    1.00000000   13.00000000    67500
T Epoch 9       episode_return    0.93766696  0.07350261   -1.12000000    0.96000000    67500
T Epoch 9       episode_time/s    0.05501295  0.00359702    0.00847664    0.09807205    67500
T Epoch 9     steps_per_second  126.77702741  5.59780179   99.06584880  154.46561131    67500
T Epoch 9         correct_rate    0.99863704  0.03689316    0.00000000    1.00000000    67500
T Epoch 9                 time  815.65879947  0.00000000  815.65879947  815.65879947        1


     info               key          mean         std           min           max    count
---------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 9    episode_length    6.90000000  0.30000000    6.00000000    7.00000000      100
E Epoch 9    episode_return    0.94100000  0.00300000    0.94000000    0.95000000      100
E Epoch 9    episode_time/s    0.05255918  0.00278256    0.04333003    0.05786009      100
E Epoch 9  steps_per_second  131.41760457  4.56410935  120.98148506  143.99995892      100
E Epoch 9      correct_rate    1.00000000  0.00000000    1.00000000    1.00000000      100
E Epoch 9              time  820.66850353  0.00000000  820.66850353  820.66850353        1


      info                  key          mean         std           min           max    count
----------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 10               return    1.18643810  0.01014737    1.15838814    1.21193051     3000
T Epoch 10         policy_ratio    0.99935357  0.00529214    0.94385695    1.12354207     3000
T Epoch 10          policy_loss    0.00014304  0.01059118   -0.01415719    0.35860378     3000
T Epoch 10           value_loss    0.00681323  0.00875590    0.00009725    0.06651944     3000
T Epoch 10              entropy    0.09741218  0.01393548    0.05833817    0.20002121     3000
T Epoch 10                 loss    0.00160142  0.01116868   -0.01121841    0.36662784     3000
T Epoch 10            grad_norm    0.24571323  1.26154198    0.02276408   38.86073685     3000
T Epoch 10  sample_data_time/ms    0.04406921  0.05441174    0.02456596    1.89679302     3000
T Epoch 10  batch_learn_time/ms   23.43404607  0.94146069   22.22606726   29.26041232     3000
T Epoch 10       episode_length    6.92985059  0.27556421    1.00000000   10.00000000    68739
T Epoch 10       episode_return    0.93741370  0.08059122   -1.07000000    0.96000000    68739
T Epoch 10       episode_time/s    0.05476912  0.00332591    0.00641341    0.07971961    68739
T Epoch 10     steps_per_second  126.78915422  5.67249274  101.72594245  155.92333843    68739
T Epoch 10         correct_rate    0.99835610  0.04051169    0.00000000    1.00000000    68739
T Epoch 10                 time  894.67980993  0.00000000  894.67980993  894.67980993        1


      info               key          mean         std           min           max    count
----------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 10    episode_length    6.93000000  0.25514702    6.00000000    7.00000000      100
E Epoch 10    episode_return    0.94070000  0.00255147    0.94000000    0.95000000      100
E Epoch 10    episode_time/s    0.05351300  0.00289747    0.04240125    0.06189075      100
E Epoch 10  steps_per_second  129.69635934  4.88310738  113.10251624  143.04573826      100
E Epoch 10      correct_rate    1.00000000  0.00000000    1.00000000    1.00000000      100
E Epoch 10              time  899.68925389  0.00000000  899.68925389  899.68925389        1


      info                  key          mean         std           min           max    count
----------  -------------------  ------------  ----------  ------------  ------------  -------
T Epoch 11               return    1.21549662  0.01253213    1.17290962    1.24680185     3000
T Epoch 11         policy_ratio    0.99914737  0.03420919    0.89613646    2.38338590     3000
T Epoch 11          policy_loss   -0.00087641  0.01389569   -0.01690255    0.49921596     3000
T Epoch 11           value_loss    0.02072323  0.01782740    0.00011480    0.10613690     3000
T Epoch 11              entropy    0.10899983  0.02013735    0.06849500    0.22038370     3000
T Epoch 11                 loss    0.00730521  0.01530571   -0.01591208    0.50527680     3000
T Epoch 11            grad_norm    0.24492276  0.85724413    0.01786815   26.18409157     3000
T Epoch 11  sample_data_time/ms    0.04287259  0.04557216    0.02402859    1.96775002     3000
T Epoch 11  batch_learn_time/ms   23.35404128  0.86766498   22.23285707   30.04403273     3000
T Epoch 11       episode_length    6.94206608  0.30223758    3.00000000   14.00000000    68371
T Epoch 11       episode_return    0.93215471  0.12920379   -1.13000000    0.97000000    68371
T Epoch 11       episode_time/s    0.05484279  0.00339308    0.02172658    0.10706252    68371
T Epoch 11     steps_per_second  126.82452245  5.52634670   99.05436904  150.29451937    68371
T Epoch 11         correct_rate    0.99578769  0.06476549    0.00000000    1.00000000    68371
T Epoch 11                 time  973.38762741  0.00000000  973.38762741  973.38762741        1


      info               key          mean         std           min           max    count
----------  ----------------  ------------  ----------  ------------  ------------  -------
E Epoch 11    episode_length    6.96000000  0.19595918    6.00000000    7.00000000      100
E Epoch 11    episode_return    0.94040000  0.00195959    0.94000000    0.95000000      100
E Epoch 11    episode_time/s    0.05410320  0.00220026    0.04149026    0.05840066      100
E Epoch 11  steps_per_second  128.74281885  3.20795600  119.86166902  144.61224043      100
E Epoch 11      correct_rate    1.00000000  0.00000000    1.00000000    1.00000000      100
E Epoch 11              time  977.39625777  0.00000000  977.39625777  977.39625777        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 12               return     1.25856077  0.01098938     1.22708869     1.28224707     3000
T Epoch 12         policy_ratio     0.99878303  0.00663642     0.89999437     1.19713748     3000
T Epoch 12          policy_loss    -0.00005917  0.00711611    -0.01515451     0.13579762     3000
T Epoch 12           value_loss     0.00582617  0.00861138     0.00009940     0.06324533     3000
T Epoch 12              entropy     0.09095858  0.01418925     0.05551084     0.19535500     3000
T Epoch 12                 loss     0.00103474  0.00758524    -0.01240870     0.13950279     3000
T Epoch 12            grad_norm     0.18712216  0.46413847     0.01251126    14.19488811     3000
T Epoch 12  sample_data_time/ms     0.04263153  0.04248770     0.02469495     1.74414599     3000
T Epoch 12  batch_learn_time/ms    23.35948205  0.87857576    22.18728093    29.83561996     3000
T Epoch 12       episode_length     6.93412465  0.30408560     3.00000000    25.00000000    67567
T Epoch 12       episode_return     0.93799473  0.07287397    -1.11000000     0.96000000    67567
T Epoch 12       episode_time/s     0.05481560  0.00350770     0.02125408     0.20795515    67567
T Epoch 12     steps_per_second   126.76639374  5.75971696    94.91807885   147.34220029    67567
T Epoch 12         correct_rate     0.99866799  0.03647241     0.00000000     1.00000000    67567
T Epoch 12                 time  1051.13447954  0.00000000  1051.13447954  1051.13447954        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 12    episode_length     6.91000000  0.28618176     6.00000000     7.00000000      100
E Epoch 12    episode_return     0.94090000  0.00286182     0.94000000     0.95000000      100
E Epoch 12    episode_time/s     0.05329423  0.00282254     0.04389880     0.05981908      100
E Epoch 12  steps_per_second   129.80324867  4.44443401   117.01951965   146.24458495      100
E Epoch 12      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 12              time  1055.14403109  0.00000000  1055.14403109  1055.14403109        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 13               return     1.29270055  0.01124281     1.25969124     1.31852651     3000
T Epoch 13         policy_ratio     0.99904058  0.00596153     0.96141905     1.13437891     3000
T Epoch 13          policy_loss     0.00018382  0.01111540    -0.01541494     0.22868384     3000
T Epoch 13           value_loss     0.00814737  0.01105441     0.00011417     0.06258956     3000
T Epoch 13              entropy     0.08896026  0.01388739     0.05328473     0.16609320     3000
T Epoch 13                 loss     0.00247830  0.01162803    -0.01168937     0.23269746     3000
T Epoch 13            grad_norm     0.17512562  0.56203785     0.00862169    16.31402779     3000
T Epoch 13  sample_data_time/ms     0.04486555  0.06270181     0.02401602     1.83766009     3000
T Epoch 13  batch_learn_time/ms    23.38818858  0.87503605    22.24164596    29.78776302     3000
T Epoch 13       episode_length     6.93281665  0.28211234     1.00000000    19.00000000    67740
T Epoch 13       episode_return     0.93766032  0.07730313    -1.18000000     0.96000000    67740
T Epoch 13       episode_time/s     0.05477681  0.00334401     0.00880016     0.14979726    67740
T Epoch 13     steps_per_second   126.82207805  5.64733665   100.32337666   147.63680933    67740
T Epoch 13         correct_rate     0.99849424  0.03877486     0.00000000     1.00000000    67740
T Epoch 13                 time  1129.02270908  0.00000000  1129.02270908  1129.02270908        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 13    episode_length     6.94000000  0.23748684     6.00000000     7.00000000      100
E Epoch 13    episode_return     0.94060000  0.00237487     0.94000000     0.95000000      100
E Epoch 13    episode_time/s     0.05377109  0.00221274     0.04547659     0.05629347      100
E Epoch 13  steps_per_second   129.13977348  3.27401581   123.85673298   143.60458121      100
E Epoch 13      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 13              time  1134.03151807  0.00000000  1134.03151807  1134.03151807        1


      info                  key           mean         std            min            max    count
----------  -------------------  -------------  ----------  -------------  -------------  -------
T Epoch 14               return     1.19641998  0.18179157     0.76692176     1.34165442     3000
T Epoch 14         policy_ratio     1.01308350  0.96224282     0.83819807    26.95873451     3000
T Epoch 14          policy_loss     0.00072625  0.01175922    -0.02593141     0.12104566     3000
T Epoch 14           value_loss     0.26902475  0.37382006     0.00009282     1.14290977     3000
T Epoch 14              entropy     0.10203693  0.04033871     0.04913829     0.29619622     3000
T Epoch 14                 loss     0.13319789  0.18663294    -0.01018986     0.56468916     3000
T Epoch 14            grad_norm     0.26462470  0.29845117     0.00700400     7.30803204     3000
T Epoch 14  sample_data_time/ms     0.04514609  0.10009262     0.02473500     4.17958107     3000
T Epoch 14  batch_learn_time/ms    23.36518139  0.85470278    22.12113608    30.22205690     3000
T Epoch 14       episode_length     6.87754187  0.46256631     2.00000000    11.00000000    69142
T Epoch 14       episode_return     0.83075627  0.45477786    -1.07000000     0.96000000    69142
T Epoch 14       episode_time/s     0.05430359  0.00441225     0.01513888     0.08900477    69142
T Epoch 14     steps_per_second   126.91018138  5.71668960    99.16891039   146.65206170    69142
T Epoch 14         correct_rate     0.94476584  0.22843674     0.00000000     1.00000000    69142
T Epoch 14                 time  1207.84144201  0.00000000  1207.84144201  1207.84144201        1


      info               key           mean         std            min            max    count
----------  ----------------  -------------  ----------  -------------  -------------  -------
E Epoch 14    episode_length     6.98000000  0.14000000     6.00000000     7.00000000      100
E Epoch 14    episode_return     0.94020000  0.00140000     0.94000000     0.95000000      100
E Epoch 14    episode_time/s     0.05327944  0.00206158     0.04555975     0.05714984      100
E Epoch 14  steps_per_second   131.15276048  4.40920625   122.48503920   142.65530665      100
E Epoch 14      correct_rate     1.00000000  0.00000000     1.00000000     1.00000000      100
E Epoch 14              time  1211.84958746  0.00000000  1211.84958746  1211.84958746        1

```

