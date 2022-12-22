## RLLib Setup and Experiments

Here we show how to launch an experiments in RLlib

First, we assume the ```conda``` is already installed, and we provide a script to install all the depedencies using ```conda```. 

```
$ cd ${GIT_ROOT}/src/rllib
$ bash deploy_conda_rllib.sh
```


To run the training

```
$ cd ${GIT_ROOT}/src/rllib
$ python run_gym_rllib.py
```

To stop the training, just do ```Ctrl+C```, a checkpoint will be saved at default location in

```
~/ray_results
```

To view the training processes in realtime, RLLib provides [tensorboard](https://tensorboard.dev) support. To launch tensorboard

```
$ tensorboard --logdir=~/ray_results/
```

and open the browser, by default, the url is ```localhost:6006```.


To replay the checkpoint, do

```
$ cd ${GIT_ROOT}/src/rllib
$ python replay_checkpoint.py <path_to_the_checkpoint>
```

More documentation can be found at [docs](docs).

