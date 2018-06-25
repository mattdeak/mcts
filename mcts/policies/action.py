class MostVisited:
    """Chooses the most visited child node of a given node."""

    def __call__(self, node):
        best_action = list(node.children)[0]
        most_visited = 0

        for action, child in self.children.items():
            if child.n > most_visisted:
                best_action = action
                most_visisted = child.n

        return best_action