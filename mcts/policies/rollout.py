import numpy as np

class RandomUnvisited:

    def __init__(self):
        pass

    def __call__(self, node, environment):
        return np.random.choice(environment.valid_actions)