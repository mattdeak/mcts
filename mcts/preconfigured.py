from .policies.action import MostVisited
from .policies.expansion import VanillaExpansion
from .policies.rollout import RandomUnvisited, RandomChoice
from .policies.selection import UCB1
from .policies.simulation import RandomToEnd
from .policies.update import VanillaUpdate
from .mcts import MCTS


def load_vanilla_mcts(environment, c=1.41, calculation_time=5):
    """Loads a vanilla MCTS object for gameplay.
    
    Keyword Arguments:
        c {float} -- [description] (default: {1.41})
    
    Returns:
        [mcts.MCTS] -- A configured MCTS object.
    """
    config = {
        "action": "most-visited",
        "expansion": "vanilla",
        "expansion_rollout": "random-unvisited",
        "simulation": "random-to-end",
        "selection": "ucb1",
        "selection_kwargs": {"C": c},
        "update": "vanilla",
    }

    mcts = MCTS(environment, calculation_time=calculation_time)
    mcts.build(config)
    return mcts


def load_nn_mcts(self, environment, model, c=1.41, calculation_time=5):
    """[summary]
    
    Arguments:
        environment {[type]} -- [description]
        model {[type]} -- [description]
    
    Keyword Arguments:
        c {float} -- [description] (default: {1.41})
        calculation_time {int} -- [description] (default: {5})
    """
