import logwood
from abc import ABC, abstractmethod

class BasePolicy:

    def __init__(self):
        self._logger = logwood.get_logger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, node, environment):
        """Call the policy"""