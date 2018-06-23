class UCB1:
    """UCB1 Policy Class. Works in the selection phase to balance
    exploration and exploitation."""
    def __init__(self, C=1.41):
        self.C = C

    def __call__(self, node, environment):

        actions = environment.valid_actions

        log_n = np.log(sum([node[action].n + epsilon for action in actions]))
        ucb1_values = [[action, node[action].value + self.C*np.sqrt(log_n/(node[action].n + epsilon))] for action in actions]
        action, value = max(ucb1_values, key=lambda x: x[1])
        return action


class PUCT:

    def __init__(self, C, prior_func):
        self.C = C
        self.prior_func = prior_func

    def __call__(self, node, environment):
        children = node.children
        priors = prior_func(environment.state)

        root_N = np.sqrt(np.sum([edge.n for edge in children.values()]))
        qus = np.zeros(len(priors))
        for i in range(len(priors)):
            prior = priors[i]
            edge = node[i]

            qu = edge.value + C * prior * root_N / (edge.n + 1e-5)
            qus[i] = qu

        return qus