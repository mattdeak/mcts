import xxhash
import logwood
import numpy as np
from utils import random_rollout, ucb1_selection
import datetime
from copy import deepcopy


class MCTNode:
    def __init__(self, state_id, state):
        self.id = state_id # Used to identify state
        self.state = state
        self.visit_count = 1
        self.win_count = 0
        self.edges = {}

    @property
    def value(self):
        return self.win_count / self.visit_count

    def best_child(self):
        best_action = list(self.edges)[0]
        best_value = 0
        for action, node in self.edges.items():
            if node.value > best_value:
                best_action = action

        return best_action

    def is_leaf(self):
        return self.edges == {}

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

class WinNode(MCTNode):

    def __init__(self):
        super().__init__('win', 'terminal')

    @property
    def value(self):
        return 1

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

class LossNode(MCTNode):
    def __init__(self):
        super().__init__('loss', 'terminal')

    @property
    def value(self):
        return 0

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

class MCTS:

    def __init__(self, environment, adversarial=True, C=1.41, calculation_time=5):
        self.environment = environment
        self.adversarial = adversarial
        self._logger = logwood.get_logger(self.__class__.__name__)
        self.calculation_time = datetime.timedelta(seconds=calculation_time)
        self.C = C
        self.nodes = {'win' : WinNode(), 'loss' : LossNode()}

        self.rollout_function = random_rollout
        self.terminal = False

        self.simulation_stats = {'Games' : 0, 'Wins' : 0, 'Losses': 0}

        self.reset_environment()

    def reset_environment(self):
        self.environment.reset()
        state = self.environment.state_space
        self.current = self._get_node(state)
        self.terminal = False

    @property
    def confidence(self):
        current_state = self.environment.state_space
        node = self._get_node(current_state)
        return node.best_child().value


    def next_action(self):
        state = self.environment.state_space
        self.current = self._get_node(state)
        if self.terminal:
            raise ValueError("Game environment is terminal. Cannot take action.")
        begin = datetime.datetime.utcnow()

        # Run MCTS for the calculation window
        games_played = 0
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.run(self.current)
            games_played += 1

        self._logger.info(f'Games Simulated: {games_played}')

        self._logger.debug('Choosing Action')
        action = self.current.best_child()
        self._logger.info(f"Action Chosen {action}")
        _, reward, done = self._take_action(self.environment, self.current, action)
        if done:
            if reward == 1:
                print("Won Game")
            else:
                print("Lost Game")
            self.terminal = True

    def run(self, root):
        clone_env = self.environment.clone()
        history = []
        history.append(root.id)
        current = root
        done = False
        self._logger.debug("Entering Selection Phase")
        while not done and current.is_leaf():
            action = self._select(current, clone_env)
            current, reward, done = self._take_action(clone_env, current, action)
            if current:
                history.append(current.id)

        self._logger.debug("Entering expansion phase")
        if not done:
            action = self._expand(clone_env)
            current, reward, done = self._take_action(clone_env, current, action)
            if current:
                history.append(current.id)

        self._logger.debug("Entering simulation phase")
        if not done:
            current, reward, done = self._simulate(current, clone_env)

        if reward == 1:
            self.simulation_stats['Wins'] += 1
        else:
            self.simulation_stats['Losses'] += 1

        self._logger.debug("Entering update phase")
        self._update(reward, history)


    def _select(self, current, env):
        """Selects an action and returns the """
        actions = env.action_space
        if all(current.edges.get(action) for action in actions):
            self._logger.debug("Calling ucb1 selection")
            action = ucb1_selection(current, C=self.C)
        else:
            action = self.rollout_function(actions)
        return action

    def _expand(self, env):
        action = self.rollout_function(env.action_space)
        return action


    def _simulate(self, current, env):
        done = False
        while not done:
            action = self.rollout_function(env.action_space)
            current, reward, done = self._take_action(env, current, action, add_edge=False)
        return current, reward, done

    def _update(self, reward, history):
        win_next_node = reward == 1
        for node_id in reversed(history):
            node = self.nodes[node_id]
            node.visit_count += 1
            if self.adversarial:
                if win_next_node:
                    self._logger.debug(f'Updating win count on id {node_id}')
                    node.win_count += 1
                # Every other node belongs to the same player
                win_next_node = not win_next_node

    def _take_action(self, env, current, action, add_edge=True):
        """Takes the action in the environment and records

        Returns: next_node, reward, done
        """
        next_node = None
        observation, reward, done = env.step(action)
        if not done:
            next_node = self._get_node(observation, add_edge=add_edge)
            if add_edge and not current.edges.get(action):
                current.edges[action] = self.nodes[next_node.id]
        else:
            if reward == 1:
                current.edges[action] = self.nodes['win']
            else:
                current.edges[action] = self.nodes['loss']
        return next_node, reward, done



    def _get_node(self, state, add_edge=True):
        """Grabs the appropriate node for the given state or generates a new one"""
        unique_id = xxhash.xxh64(state).digest()
        if unique_id in self.nodes:
            return self.nodes[unique_id]
        else:
            node = MCTNode(unique_id, state)
            if add_edge:
                self.nodes[unique_id] = deepcopy(node)
            return node
