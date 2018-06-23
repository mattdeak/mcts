import xxhash
import logwood
import numpy as np
import datetime
from sortedcontainers.sorteddict import SortedDict
from copy import deepcopy


class MCTNode:
    """Node class for MCTS. Tracks state and win information."""
    def __init__(self, state_id, state, player_turn):
        self.id = state_id # Used to identify state
        self.state = state
        self.player = player_turn
        self.n = 0
        self.w = 0
        self.children = SortedDict()

    def add_child(self, action, child):
        self.children[action] == child

class NodeTree:
    def __init__(self):
        
    def add_node(self, parent_id, action, node):
        """Adds a node to the node tree"""
        

    def delete_node(self, node_id):
        """Deletes a node from the node tree"""

    def get_by_id(self, node_id):
        """Retrieves a node by the node ID"""

    def get_by_node(self, node):
        """Retrieves the node"""

    def reset(self):
        self.nodes = {}



class MCTS:
    def __init__(self, environment, action_policy, selection_policy, simulation_policy,
                update_policy):
        self.environment = environment
        self.action_policy = action_policy
        self.selection_policy = selection_policy
        self.simulation_policy = simulation_policy
        self.update_policy = update_policy

        self.terminal = False
        self.game_history = []

    
    def act(self):
        state = self.environment.state
        player = self.environment.last_player
        self.current = 


    def run(self, root):
        env_clone = self.environment.clone()
        history = []
        history.append(root.id)
        current = root
        done = False

        # Selection Phase
        while not done and not current.is_leaf():
            action = self.selection_policy.act(current, env_clone)
            self.history.append(current.id)

        # Expansion Phase
        if not done:
            action = self.expansion_policy.act(current, env_clone)

        # Simulation Phase
        if not done:
            current, reward, done = self.simulation_policy.act(current, clone_env)

        # Update Phase
        self.update_policy.update(current, env_clone, winner, reward, history)