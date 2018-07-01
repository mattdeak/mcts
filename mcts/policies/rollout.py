from ..base.policy import BasePolicy
import numpy as np

class RandomChoice(BasePolicy):

    def __call__(self, node, environment):
        return np.random.choice(environment.actions)

class RandomUnvisited(BasePolicy):

    def __call__(self, node, environment):
        
        if node.expanded:
            unvisited_actions = [a for a, edge in node.edges.items() if edge.n == 0]
            return np.random.choice(
                np.intersect1d(unvisited_actions, environment.actions)
                )
        # If the node is a leaf node
        else:
            return environment.actions