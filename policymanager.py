from abc import ABC, abstractmethod
from utils import ucb1
from numpy.random import choice
class BasePolicyManager(ABC):
    """
    The base class for the MCTS policy manager.

    By inheriting this class and implementing the policy
    methods, it should be quite simple to build a policy using neural networks
    or other heuristic methods.
    """

    @abstractmethod
    def rollout(self, state):
        pass

    @abstractmethod
    def selection(self, node):
        pass

class DefaultPolicyManager(BasePolicyManager):
    """
    The default policy manager for MCTS.

    An MCTS object using this policy manager will use random rollout and
    UCB1 selection.
    """
    def __init__(self, C=1.41):
        self.C = C

    def rollout(self, env):
        return choice(env.actions)

    def selection(self, node):
        return ucb1(node)
