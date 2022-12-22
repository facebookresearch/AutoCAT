# Deployment of training on a new cluster

The RL training requires rllib and lots of other dependencies. The best way to resolve it is to use an conda environment. We have preparared a yaml file to create the conda environment. Assuming ```conda``` is already installed the cluster.

The following creates teh conda environment ```rllib```

```
$ cd {path-to-CacheSimulator}/src
$ conda activate base
$ conda-env create -n rllib -f=rllib.yml
```

Then activate the environment
```
$ conda activate rllib 
```

Now, depdending on the configuration of the cluster and how much resource you want to use, edit ```num_gpus``` and ```num_workers``` in the ```src/test_custom_policy_diversity_works.py``` file.

Then launch the training in 

```
(rllib) $ python run_gym_rllib_agent_balcklist.py
```
To start training.

To monitor the training progress, launch ```tensorboard``` using the following command (in MSCode)

```
(rllib) $ tensorboard --logdir=~/ray_results/
```

Which posts to ```https://localhost:6006``` (on MSCode, port ```6006``` on the server is automatically forwarded to your client computer, so that you can directly look at the result on a local browser ```localhost:6006```).