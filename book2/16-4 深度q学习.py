import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import gym
import numpy as np

env = gym.make("MsPacman-v0")
obs = env.reset()
print(obs.shape)