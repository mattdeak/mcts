from ..base.policy import BasePolicy
import numpy as np

class UCB1(BasePolicy):
    """UCB1 Policy Class. Works in the selection phase to balance
    exploration and exploitation."""
    def __init__(self, C=1.41):
        self.C = C
        super().__init__()

    def __call__(self, node, environment):

        actions = environment.actions
        epsilon = 1e-5

        log_n = np.log(sum([node[action].n + epsilon for action in actions]))
        ucb1_values = [[action, node[action].q + self.C*np.sqrt(log_n/(node[action].n + epsilon))] for action in actions]
        action, value = max(ucb1_values, key=lambda x: x[1])
        return action


class PUCT(BasePolicy):

    def __init__(self, C=1.41):
        self.C = C
        super().__init__()

    def __call__(self, node, environment):
        edges = node.edges
        priors = [action.p for action in children.values()]

        root_N = np.sqrt(np.sum([edge.n for edge in children.values()]))
        qus = np.zeros(len(priors))
        for i in range(len(priors)):
            prior = priors[i]
            edge = node[i]

            qu = edge.value + C * prior * root_N / (edge.n + 1e-5)
            qus[i] = qu

        return qus