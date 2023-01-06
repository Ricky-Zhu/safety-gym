# How to use the randomizable Safe Doggo env

Usage: 

the density_front correspond to the density of the front body of the doggo

```python
import gym, safety_gym

env = gym.make('RandomizeSafexp-DoggoGoal1-v0')

## to randomize the parameters within the distribution (gaussian)
front_density_mean, front_density_var = 0.5, 0.1
rear_density_mean, rear_density_var = 0.5, 0.1

# mind the order of the parameters
parameters = [front_density_mean, rear_density_mean, front_density_var, rear_density_var]
env.set_values(parameters)

## the normal reset which doesn't change the current parameter values
env.reset()
```

