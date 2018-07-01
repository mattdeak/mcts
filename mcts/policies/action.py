from ..base.policy import BasePolicy
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

    # TODO: Implement this
    def __call__(self, node):
        pass
