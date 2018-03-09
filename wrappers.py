from copy import deepcopy
import numpy as np
import gym

class GymWrapper:

    def __init__(self, env):
        if type(env.action_space) != gym.spaces.discrete.Discrete:
            raise NotImplementedError("Only environments with discrete action spaces are supported")

        self.env = env
        self._observation = np.array(self.env.reset())
        self.action_space = np.arange(self.env.action_space.n)
        self.env.reset()

    @property
    def state_space(self):
        self._observation


    def step(self, action):
        self._observation, reward, done, _ = self.env.step(action)
        return self._observation, reward, done


    def clone(self):
        return deepcopy(self.env)
