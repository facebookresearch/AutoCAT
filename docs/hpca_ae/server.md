# Server access

We provide the following server for experiments. 

```
server name:  fukushima.ece.vt.edu
jump host: ssh.ece.vt.edu
```

All software for the experiments should already be installed on the machine. 
We provide three accounts ```hpca23ae01```, ```hpca23ae02```, ```hpca23ae03``` to access these machines. Reviewers can coordinate with each other on who to use which accounts, one way is to sort according to the aphabetically order of the last name of the reviewers.


For each account, please use the following command to access each of them.

## account 1
```
$ssh hpca23ae01@fukushima.ece.vt.edu -J hpca23ae01@ssh.ece.vt.edu
```
Then input the following password twice when prompt.
```
DVKwn5s83v2ejkp
```

## account 2
```
$ssh hpca23ae02@fukushima.ece.vt.edu -J hpca23ae02@ssh.ece.vt.edu
```
Then input the following password twice when prompt.
```
J6SBZjf9v3axrhe
```

## account 3
```
$ssh hpca23ae03@fukushima.ece.vt.edu -J hpca23ae03@ssh.ece.vt.edu
```
Then input the following password twice when prompt.
```
HL543kxt9EJen3W
```
## activate conda env

After log onto the server, use 
```
$ conda activate py38
```
To activate the conda environment, all the AutoCAT training can be performed under this environment.

Disclaimer: You are only allowed to use the machine and network as intended for the artifact evaluation HPCA23 AE #1. Every data on the accessed machines has to be considered confidential. All software for the experiments should already be installed on the machine. However, if something is missing, you are allowed to install it. If there are any questions or problems, we are happy to help! These accounts will be decommissioned after the AE period.