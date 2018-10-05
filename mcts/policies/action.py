import numpy as np
from ..base.policy import BasePolicy
from ..utils import node_to_probability_distribution

class MostVisited(BasePolicy):
    """Chooses the most visited child node of a given node."""
    def __call__(self, node):
        edges = node.edges
        best_action = list(edges.keys())[0]
        most_visited = 0

        for action, child in edges.items():
            if child.n > most_visited:
                best_action = action
                most_visited = child.n

        return best_action

class ProportionalToVisitCount(BasePolicy):
    """Probabilistically chooses an action with a likelihood that is
    proportional to that nodes visit count."""
    def __init__(self, t=0.1):
        self.t = t

    def __call__(self, node):
        """Selects an action probabilistically from a set of node edges.
        Action selection is proportional to N(A)^(1/t)/sum(N(B)^1/t)

        Arguments:
            node {mcts.tree.Node} -- The node to process into search probabilities.
        
        Returns:
            int -- The selected action
        """
        search_array = node_to_probability_distribution(node, t=self.t)
        return np.random.choice(
            search_array[:, 0], 
            p=search_array[:, 1]
            ).astype(np.int)
