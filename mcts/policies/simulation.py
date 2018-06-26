from ..base.policy import BasePolicy

class SimulateToEnd(BasePolicy):

    def __init__(self, rollout_policy):
        self.rollout = rollout_policy
        super().__init__()

    def __call__(self, current, environment):
        action = self.rollout(current, environment)
        observation, reward, done = environment.step(action)
        while not done:
            action = self.rollout(current, environment)
            observation, reward, done = environment.step(action)
        
        return observation, reward, done