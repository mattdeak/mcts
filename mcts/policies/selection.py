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


class PUCT(BasePolicy):
    """UCB1 Policy Class. Chooses an action probabalistically
    based on the priors and action-values of edges in a node."""
    def __init__(self, C=1.41):
        self.C = C
        super().__init__()

    def __call__(self, node):
        edges = node.edges

        root_N = np.sqrt(np.sum([edge.n for edge in edges.values()]))
        qus = {}

        for action, edge in edges.items():

            qu = edge.q + self.C * edge.p * root_N / (edge.n + 1)
            # A little confusing to use the qu value as the key,
            # But this way is slightly higher performance for our task
            qus[qu] = action

        return qus[max(qus)]

