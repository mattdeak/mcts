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