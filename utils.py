import numpy as np

def ucb1(node, C=1.41):
    """Returns an action that maximizes the UCB1 formula"""
    edges = node.edges
    log_visit_count = np.log(sum([edge.visit_count for edge in edges.values()]))

    ucb1_values = [[action, edge.value + C*np.sqrt(log_visit_count/edge.visit_count)] for action, edge in edges.items()]
    action, value = max(ucb1_values, key=lambda x: x[1])
    return action

def puct(node, priors, C=1.41):
    """Returns the action that maximizes the AlphaZero PUCT variation.

    Input:
        node (MCTNpde): The node to select actions from.
        priors (list or array): A list of priors where the index of the prior
                represents its associated action
        C (float): A hyperparameter to control exploration/exploitation."""
    edges = node.edges
    root_N = np.sqrt(np.sum([edge.visit_count for edge in edges.values()]))
    qus = []
    for i in range(priors):
        prior = priors[i]
        edge = node[i]

        qu = C * prior * root_N / edge.visit_count
        qus.append(qu)

    return np.argmax(qus)
