import logwood
from abc import ABC, abstractmethod

class BasePolicy(ABC):

    def __init__(self):
        self._logger = logwood.get_logger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, node, environment):
        """Call the policy"""

class NodeTrackingPolicy(BasePolicy):

    def add_tree(self, tree):
        """Adds the MCTS node-tree for internal reference."""
        self.tree = tree