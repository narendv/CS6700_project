import os
import gym
import gym_bellman
import numpy as np
from tqdm import tqdm

config = {"acrobot": [0, 0], "taxi": [0, 0], "kbca": [0, 0], "kbcb": [0, 0], "kbcc": [0, 0]}

env = gym.make("Acrobot-v1")
config['acrobot'] = [env.observation_space.shape[0],env.action_space.n]

env = gym.make("Taxi-v3")
config['taxi'] = [env.observation_space.n,env.action_space.n]

env = gym.make("gym_bellman:kbc-a-v0")
config['kbca'] = [env.N+1,env.action_space.n]
env = gym.make("gym_bellman:kbc-b-v0")
config['kbcb'] = [env.N+1,env.action_space.n]
env = gym.make("gym_bellman:kbc-c-v0")
config['kbcc'] = [env.N+1,env.action_space.n]