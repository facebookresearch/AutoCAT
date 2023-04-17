AutoCAT
==============
This repo contains artifacts of the paper:

* "AutoCAT: Reinforcement Learning for Automated Exploration of Cache-Timing Attacks" (HPCA'23).

The paper will be available at the [HPCA website](https://hpca-conf.org/2023/).

## Artifact contents

The artifact contains two parts

* [Cache Gym Environment](src/cache_guessing_game_env_impl.py) and PPO trainer

    * Cache Gym Environment is based on an [open source CacheSimulator](https://github.com/auxiliary/CacheSimulator) from [auxiliary](https://github.com/auxiliary).
    * PPO trainer is using [rlmeta](https://github.com/facebookresearch/rlmeta) from [Meta AI](https://ai.facebook.com).

* [StealthyStreamline Attack code](src/stealthy_streamline)

## System requirement

The reinforcement learning is performed on Nvidia GPU. We require proper CUDA support (version>10.2) on the machine. To check the GPU and Cuda version, use ```nvidia-smi``` command, and the output should look like this.

```
Sun Oct 23 20:43:01 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla M60           On   | 00000000:00:1E.0 Off |                    0 |
| N/A   27C    P8    14W / 150W |      0MiB /  7680MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Set up Enviroment on a GPU machine

We use conda to manage all the python dependencies, we assume the ```conda``` is already installed, and we provide a script to install all the depedencies using ```conda```.

Creating a conda environment:

```
$ conda create --name py38 python=3.8
```
Then press enter when prompt.

Activate the conda environment

```
$ conda activate py38
```
Undet the py38 environment

```
(py38) $ pip install scikit-learn seaborn pyyaml hydra-core terminaltables pep517
```

The environment is based on openai [gym](https://github.com/openai/gym). To install it, use the following.

```
(py38) $ pip install gym
```

Please follow the [PyTorch Get Started](https://pytorch.org/get-started/locally/) website to install PyTorch with proper CUDA version. One example is listed below.
```
(py38) $ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

The enviroment needs [moolib](https://github.com/facebookresearch/moolib) as the RPC backend for distributed RL training. Please follow the moolib installation instructions.
We recommend building moolib from source with the following steps.

```
(py38) $ git clone https://github.com/facebookresearch/moolib
(py38) $ cd moolib
(py38) $ git submodule sync && git submodule update --init --recursive
(py38) $ pip install .
```

The RL trainer is based on [RLMeta](https://github.com/facebookresearch/rlmeta) at commit 1057fbbf2637a002296afe5071e6ac0e7b630fe0.

Please follow setup process on [rlmeta](https://github.com/facebookresearch/rlmeta) for install RLMeta.

```
(py38) $ git clone https://github.com/facebookresearch/rlmeta
(py38) $ cd rlmeta
(py38) $ git checkout 1057fbbf2637a002296afe5071e6ac0e7b630fe0
(py38) $ git submodule sync && git submodule update --init --recursive
(py38) $ pip install -e .
```

Alternatively, we have prebuilt a docker image that can be deployed on a AWS g5.xlarge instance with Deep Learning AMI GPU PyTorch 2.0.0 (Ubuntu 20.04) 20230401 image (ami-0a4caa099fc23090f).
Please follow [this](https://docs.docker.com/engine/install/ubuntu/) on installing the latest version of docker engine.
The docker image can be pulled by 

```
$ docker pull ml2558/autocat-rlmeta
```


Run the docker image

```
$ docker run -it --gpus all ml2558/autocat-rlmeta /bin/bash
```

Set the conda environment variables and activate ```py38``` environment.

```
$ eval "$(/root/miniconda3/bin/conda shell.bash hook)" 
$ conda activate py38
```

This will prepare all dependencies for running the experiments.


## General flow for Training and Evaluating RL agent

Once the system is set up. Please checkout our code.

```
(py38) $ git clone https://github.com/facebookresearch/AutoCAT
(py38) $ cd AutoCAT
```

Then, set the path to the ```AutoCAT``` repo.

```
$ export GIT_ROOT=<path_to_the_autocat_repo>
```

You can launch the experiment to train the RL agent. One basic example is shown below. We provide the training scripts with parameters we are using on our machines in the ```src/rlmeta/data/``` dir for each of the experiments.

```
$ cd ${GIT_ROOT}/src/rlmeta
$ python train_ppo_attack.py
```

At the beginning the replay buffer will be filled and the print out will look like.
```
...
[20:53:45] Warming up replay buffer: [   1478 / 131072 ]                                         replay_buffer.py:208
[20:53:46] Warming up replay buffer: [   1709 / 131072 ]                                         replay_buffer.py:208
[20:53:47] Warming up replay buffer: [   1937 / 131072 ]                                         replay_buffer.py:208
[20:53:48] Warming up replay buffer: [   2179 / 131072 ]                                         replay_buffer.py:208
[20:53:54] Warming up replay buffer: [   3596 / 131072 ]                                         replay_buffer.py:208
...
```

After the replay buffer is filled, the training logs will be like

```
[20:58:31] Training for num_steps = 3000                                                             ppo_agent.py:144
Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
[2022-10-23 21:04:06,939][root][INFO] - {"return": {"mean": -7.837982208490361, "std": 11.00719994082304, "min": -70.12750244140625, "max": -1.0114586353302002, "count": 3000, "key": "return"}, "policy_ratio": {"mean": 0.9995904984275495, "std": 0.01008690819091325, "min": 0.9565219879150391, "max": 1.0483020544052124, "count": 3000, "key": "policy_ratio"}, "policy_loss": {"mean": -0.015300600946259997, "std": 0.013428813022190391, "min": -0.051137540489435196, "max": 0.050006382167339325, "count": 3000, "key": "policy_loss"}, "value_loss": {"mean": 64041.523216316404, "std": 105816.17287373912, "min": 1.1493895053863525, "max": 609683.625, "count": 3000, "key": "value_loss"}, "entropy": {"mean": 2.07833841276169, "std": 0.012247777578129993, "min": 2.0188612937927246, "max": 2.113884449005127, "count": 3000, "key": "entropy"}, "loss": {"mean": 32020.725061843725, "std": 52908.095275196785, "min": 0.5378807187080383, "max": 304841.78125, "count": 3000, "key": "loss"}, "grad_norm": {"mean": 80.6914478134663, "std": 560.7136433109384, "min": 0.14680133759975433, "max": 16948.568359375, "count": 3000, "key": "grad_norm"}, "sample_data_time/ms": {"mean": 0.33627491199513326, "std": 0.8351939263832907, "min": 0.0283070003206376, "max": 10.895442000219191, "count": 3000, "key": "sample_data_time/ms"}, "batch_learn_time/ms": {"mean": 110.58090069833055, "std": 6.249485072195114, "min": 105.28848000012658, "max": 426.97272399982467, "count": 3000, "key": "batch_learn_time/ms"}, "episode_length": {"mean": 2.4042961170589154, "std": 1.8968691140142928, "min": 1.0, "max": 26.0, "count": 43807, "key": "episode_length"}, "episode_return": {"mean": -0.8572643641427186, "std": 0.5330088683162034, "min": -1.25, "max": 0.99, "count": 43807, "key": "episode_return"}, "episode_time/s": {"mean": 0.013489203685554783, "std": 0.015033721112400809, "min": 0.004837646999476419, "max": 2.085678069000096, "count": 43807, "key": "episode_time/s"}, "steps_per_second": {"mean": 189.22392526112355, "std": 61.580304516542384, "min": 1.4383811406897724, "max": 326.53334890685426, "count": 43807, "key": "steps_per_second"}, "correct_rate": {"mean": 0.07838929851393629, "std": 0.26878321449158576, "min": 0.0, "max": 1.0, "count": 43807, "key": "correct_rate"}, "info": "T Epoch 0", "phase": "Train", "epoch": 0, "time": 641.0643177460006}
```
Look at the ```correct_rate``` and ```episode_return``` for training progress. If the ```correct_rate``` is close to ```1.0```, this means the attack has high success rate.

If the there are no errors reported 
Use ```Ctrl+C``` to stop the training, which will save the checkpoint of the RL agent to the following path ```${GIT_ROOT}/src/rlmeta/outputs/${DATE}/${TIME}```

To extract the pattern of the RL agent, use the following script

```
$ cd ${GIT_ROOT}/src/rlmeta
$ python sample_attack.py env_config=hpca_ae_exp_4_1 checkpoint=${GIT_ROOT}/src/rlmeta/data/table4/hpca_ae_exp_4_1/ppoagent.pth
```

For several scenarios, training may take long time, also due to the undeterministic nature of reinforcement learning, the trained results vary from each invokations. To save the time of reviewers, we provide pretrained checkpoints and reviewers can sample it directly.

## Experiments

We provide scripts to run the following experiments appeared in the original paper.

* [Table IV: attacks found on CacheSimulator](docs/hpca_ae/table4.md)
* [Table V: RL training with different replacement policies](docs/hpca_ae/table5.md)
* [Table VI: random replacement policies](docs/hpca_ae/table6.md)
* [Table VII: comparison of PLRU with and without PLCache](docs/hpca_ae/table7.md)
* [Table VIII: bit rate, autocorrection and accuracy for attacks bypassing CChunter](docs/hpca_ae/table8.md)
* [Table IX: bit rate, guess accuarcy and detection rate for attacks bypassing SVM-based detector](docs/hpca_ae/table9.md)
* [Figure 6: measuring SteathyStreamline attack bit rate and error rate](docs/hpca_ae/figure6.md)

Notice: due to Table III in the paper depends on hardware that are not accessible now, we do not intend to reproduce Table III in this evaluation.

Please go to 

* [```${GIT_ROOT}/docs/hpca_ae/table4.md```](docs/hpca_ae/table4.md)
* [```${GIT_ROOT}/docs/hpca_ae/table5.md```](docs/hpca_ae/table5.md)
* [```${GIT_ROOT}/docs/hpca_ae/table6.md```](docs/hpca_ae/table6.md)
* [```${GIT_ROOT}/docs/hpca_ae/table7.md```](docs/hpca_ae/table7.md)
* [```${GIT_ROOT}/docs/hpca_ae/table8.md```](docs/hpca_ae/table8.md)
* [```${GIT_ROOT}/docs/hpca_ae/table9.md```](docs/hpca_ae/table9.md)
* [```${GIT_ROOT}/docs/hpca_ae/figure6.md```](docs/hpca_ae/figure6.md) 
 
for the corresponding instructions on how to run each experiments.

## Repo Structure 

```
-configs            # this is the directory for CacheSimulotor configuration
-docs               # documentations
-env_test           # contains testing suit for simulator and replacement policy
-src
 |--config          # gym environment configurations
 |--cyclone_data    # data for training cyclone svm classifier
 |--fig             # positions for storing the figure
 |--models          # customized pytorch models for the RL agent to use
 |--rllib           # scripts for launching RLLib based experiments
 |--rlmeta          # scripts for launching RLMeta-basee experiments
 |--setup_scripts   # some scripts for setup the environment
 |--cache.py        # the cache logic implementation
 |--cache_simulator.py              # the interface of the cache simulator
 |--replacement_policy.py           # define the replacement policy for the cache
 |--cache_guessing_game_env_impl.py # the gym implementation of the cache
 |--cchunter_wrapper.py             # the wrapper that implements cchunter attack detector
 |--cyclone_wrapper.py              # the wrapper that implements cyclone attack detector
-third_party        # position for third-party libraries like 
-traces             # places for traces of CacheSimulator
```
## Contact

Please direct any questions to Mulong Luo ```ml2558@cornell.edu```.

## Research Paper

The paper is available in the procceedings the 29th Sympisum on High Performance Computer Architecture [(HPCA)](https://hpca-conf.org/2023/). Please cite our work with the following bibtex entry.

```bibtex
@inproceedings{luo2023autocat
  title={{AutoCAT: Reinforcement Learning for Automated Exploration of Cache-Timing Attacks}},
  author={Mulong Luo and Wenjie Xiong and Geunbae Lee and Yueying Li and Xiaomeng Yang and Amy Zhang and Yuandong Tian and Hsien-Hsin S. Lee and G. Edward Suh},
  booktitle={29th Sympisum on High Performance Computer Architecture (HPCA)},
  year={2023}
}
```

### License

This project is under the GNU General Public License (GPL) v2.0 license. See [LICENSE](LICENSE) for details.
