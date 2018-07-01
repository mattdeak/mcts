from ..base.policy import BasePolicy
import numpy as np

class UCB1(BasePolicy):
    """UCB1 Policy Class. Works in the selection phase to balance
    exploration and exploitation for Vanilla MCTS."""
    def __init__(self, C=1.41):
        self.C = C
        super().__init__()

    def __call__(self, node):
        actions = node.edges.keys()

        log_n = np.log(sum([node[action].n for action in actions]))
        ucb1_values = [[action, node[action].q + self.C*np.sqrt(log_n/(node[action].n + 1))] for action in actions]
        action, value = max(ucb1_values, key=lambda x: x[1])
        return action


class PUCT:
    """UCB1 Policy Class. Chooses an action probabalistically
    based on the priors and action-values of edges in a node."""
    def __init__(self, C=1.41):
        self.C = C
        super().__init__()

    def __call__(self, node):
        
        edges = node.edges
        arr = np.array([[action, edge.p, edge.n, edge.q] for action, edge in edges.items()])
        
        # Softmax the priors
        arr[:, 1] = 1/(1 + np.exp(arr[:, 1]))
        # PUCT formula
        qus = arr[:, 3] + self.C * arr[:, 1] \
              * np.sqrt(np.sum(arr[:, 2])) \
              / (arr[:, 2] + 1)

        arr[qus.argmax(), 0]

        return arr[qus.argmax(), 0].astype(int)

