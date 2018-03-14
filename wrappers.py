from copy import deepcopy
import numpy as np
import gym

class GymWrapper:

    def __init__(self, env):
        if type(env.action_space) != gym.spaces.discrete.Discrete:
            raise NotImplementedError("Only environments with discrete action spaces are supported")

        self.env = env
        self._observation = np.array(self.env.reset())

    @property
    def state(self):
        return self._observation

    @property
    def actions(self):
        return np.arange(self.env.action_space.n)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        self._observation = np.array(observation)
        return self._observation, reward, done

    def reset(self):
        self.env.reset()

    def clone(self):
        return deepcopy(self)
