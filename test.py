import gym, safety_gym
import numpy as np
from safety_gym.wrappers.randomize_wrapper import RandomizeWrapper

env = gym.make('RandomizeSafexp-DoggoGoal1-v0')
config_path = '/home/ruiqi/projects/envs/safety-gym/safety_gym/randomize_configs/doggo/default.json'
env = RandomizeWrapper(env, config_path)

env.reset()
env.turn_on_randomization()
env.reset()

for i in range(1000):
    s, r, d, _ = env.step(env.action_space.sample())
    if (i+1) % 500==0:
        env.turn_off_randomization()
        env.reset()

    env.render()
