# How to calculate the correct rate

We want to calculate the correct rate for the agent. Each episode makes one guess, and in order to calculate the correct rate, we need multiple episodes (for example one epoch contains multiple episodes/steps).

For each step, the ```info``` is returned in one of three cases:

```
if done == True:
    if reward > 0:
        info = {"is_guess": True, "guess_correct": True}
    else:
        info = {"is_guess": True, "guess_correct": False}
    else:
      info = {"is_guess": False}
```

For each epoch, ```num_guess_correct``` and ```num_is_guess``` is looked up  based on the ```info``` dictionary.

Here is how the correct rate is calculated for each epoch:

```
correct_rate = num_guess_correct / num_is_guess
```