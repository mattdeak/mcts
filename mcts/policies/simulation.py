from ..base.policy import BasePolicy
from .rollout import RandomChoice

class RandomToEnd(BasePolicy):

    def __init__(self):
        self.rollout = RandomChoice()
        super().__init__()

    def __call__(self, current, environment):
        action = self.rollout(current, environment)
        observation, reward, done = environment.step(action)
        while not done:
            action = self.rollout(current, environment)
            observation, reward, done = environment.step(action)
        
        return observation, reward, done