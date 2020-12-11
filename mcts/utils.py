import numpy as np


def play_game(mcts):
    mcts.reset()
    mcts.environment.reset()
    game_history, reward, done, winner = mcts.act()
    while not done:
        game_history, reward, done, winner = mcts.act()

    return game_history, reward, winner


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.

    SOURCE: https://nolanbconaway.github.io/blog/2017/softmax-numpy
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def node_to_probability_distribution(node, t=0.1):
    """Converts a nodes edge information into a
    search probability distrubtion proportional to
    N(A)^(1/t)/sum(N(B)^1/t)
    
    Arguments:
        node {mcts.tree.Node} -- The node used to process
            search probabilities
    
    Keyword Arguments:
        t {float} -- temperature argument [0,1]. 
            Values close to 1 return more exploratory
            probabilities, values close to zero are more greedy.
            (default: {0.1})

    Returns:
        [numpy.array] -- An array of shape (n_actions, 2) where the 
        first column corresponds to the action and the second to the
        search probability of that action

    """
    edges = node.edges
    arr = np.array(
        [[np.float16(action), np.float(edge.n)] for action, edge in edges.items()]
    )
    if arr[:, 1].sum() == 0:
        raise ValueError("No visits in edges")
    arr[:, 1] = arr[:, 1] ** (1 / t) / np.sum(arr[:, 1] ** (1 / t))

    return arr
