import numpy as np

def ucb1_selection(node, C=1.41):
    """Returns an action that maximizes the UCB1 formula"""
    edges = node.edges
    visit_count = sum([edge.visit_count for edge in edges])

    ucb1_values = [[action, edge.value + C*np.sqrt(visit_count/edge.visit_count)] for action, edge in edges.items()]
    action, value = max(ucb1_values, lambda x: x[1])
    return action


def random_rollout(actions):
    return np.random.choice(actions)
