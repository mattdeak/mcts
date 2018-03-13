from abc import ABC, abstractmethod
from utils import ucb1
import numpy as np
class BasePolicyManager(ABC):

    @abstractmethod
    def rollout(self, state):
        pass

    @abstractmethod
    def selection(self, node):
        pass

class DefaultPolicyManager(BasePolicyManager):

    def __init__(self, C=1.41):
        self.C = C

    def rollout(self, env):
        return np.random.choice(env.actions)

    def selection(self, node):
        return ucb1(node)
