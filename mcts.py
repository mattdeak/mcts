import xxhash
import numpy as np
from .utils import random_rollout, ucb1

class MCTNode:

    def __init__(self, state_id, state):
        self.id = state_id # Used to identify state
        self.state = state
        self.visit_count = 1
        self.win_count = 0
        self.edges = None

    @property
    def value(self):
        return self.win_count / self.visit_count

    def is_leaf(self):
        return self.edges is None

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

class MCTS:

    def __init__(self. environment, uct_function=None, rollout_function=None, depth=50, calculation_time=5):
        self.environment = environment
        self.depth = 50
        self.calculation_time = 5

        self.root = MCTNode(environment.state_space)

        # Dictionary to keep track of nodes for fast lookup
        self.nodes = {self.root.id : self.root}

        if not uct_function:
            self.uct_function = ucb1
        else:
            self.uct_function = uct_function

        if not rollout_function:
            self.rollout_function = random_rollout
        else:
            self.rollout_function == rollout_function

    def run(self, iterations=100):
        history = []
        current = self.head
        while 
        while not current.is_leaf():
            current = self.select(current)


    def _select(self):


    def _expand(self, node):
        action = self.rollout_function(node.state)
        observation, reward, done = self.environment.step(action)
        next_node = self._get_node(observation)
        current.edges[action] = next_node
        current = next_node
        return current


    def _simulate(self, current):
        simulation_history = []
        done = False
        while not done:
            action = rollout_function(current)
            observation, reward, done = self.environment.step(action)
            current = self._get_node(observation)
            simulation_history.append(current)
        return reward

    def _update(self, reward, history):
        win_next_node = reward == 1
        for node in reversed(history):
            node.visit_count += 1
            if win_next_node:
                node.win_count += reward
            # Every other node belongs to the same player
            win_next_node = not win_next_node


    def _get_node(self, state):
        """Grabs the appropriate node for the given state or generates a new one"""
        unique_id = xxhash.xxh64(state)
        if unique_id in self.nodes:
            return self.nodes[unique_id]
        else:
            node = MTCNode(unique_id, state)
            self.nodes[unique_id] = node
            return node
