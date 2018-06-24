import numpy as np

class RandomChoice:

    def __init__(self):
        pass

    def __call__(self, node, environment):
        return np.random.choice(environment.valid_actions)

class RandomUnvisited:

    def __call__(self, node, environment):
        if node.expanded:
            unvisited_actions = [a for a in node.edges if a.n == 0]
            return np.random.choice(
                np.intersect(
                    actions, environment.valid_actions
                    )
                )
        # If the node is a leaf node
        else:
            return environment.valid_actions