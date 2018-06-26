class SimulateToEnd:

    def __init__(self, rollout_policy):
        self.rollout = rollout_policy

    def __call__(self, current, environment):
        action = self.rollout(current, environment)
        observation, reward, done = environment.step(action)
        while not done:
            action = self.rollout(current, environment)
            observation, reward, done = environment.step(action)
        
        return observation, reward, done