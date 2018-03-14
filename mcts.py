import xxhash
import logwood
import numpy as np
from utils import ucb1
from policymanager import DefaultPolicyManager
import datetime
from sortedcontainers import sorteddict
from copy import deepcopy


class MCTNode:
    """Node class for MCTS. Tracks state and win information."""
    def __init__(self, state_id, state):
        self.id = state_id # Used to identify state
        self.state = state
        self.visit_count = 0
        self.win_count = 0
        self.edges = SortedDict()

    @property
    def value(self):
        return self.win_count / self.visit_count

    def most_visited(self):
        """Returns the action that leads to the most visited child node."""
        best_action = list(self.edges)[0]
        most_visited = 0
        for action, node in self.edges.items():
            if node.visit_count > most_visited:
                best_action = action
                most_visited = node.visit_count

        return best_action

    def best_child(self):
        """Returns the best child node"""
        edges = list(self.edges.values())
        best_index = 0
        best_value = 0

        i = 0
        for action, node in self.edges.items():
            if node.value > best_value:
                best_index = i
            i += 1

        return edges[best_index]

    def is_leaf(self):
        return self.edges == {}

    def __getitem__(self, action):
        item = self._edges.get(action)
        if not item:
            item = MCTNode(1, None) # Return Dummy MCTNode for never visited

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class MCTS:

    def __init__(self, environment, policy_manager=None, adversarial=True, C=1.41, calculation_time=5):
        self.environment = environment
        self.adversarial = adversarial
        self._logger = logwood.get_logger(self.__class__.__name__)
        self._calculation_time = datetime.timedelta(seconds=calculation_time)
        self.C = C
        self.nodes = {}

        if policy_manager:
            self.policy_manager = policy_manager
        else:
            self.policy_manager = DefaultPolicyManager(C=C)

        self.terminal = False

        self.reset_environment()

    @property
    def calculation_time(self):
        return self._calculation_time

    @calculation_time.setter
    def calculation_time(self, seconds):
        self._calculation_time = datetime.timedelta(seconds=seconds)

    def reset_environment(self):
        """Resets the game environment"""
        self.environment.reset()
        state = self.environment.state
        self.current = self._get_node(state)
        self.terminal = False


    def act(self):
        """Takes the next action in the current game environment."""
        state = self.environment.state
        self.current = self._get_node(state)
        if self.terminal:
            raise ValueError("Game environment is terminal. Cannot take action.")
        begin = datetime.datetime.utcnow()

        # Run MCTS for the calculation window
        games_played = 0

        while datetime.datetime.utcnow() - begin < self._calculation_time:
            self.run(self.current)
            games_played += 1

        self._logger.info(f'Games Simulated: {games_played}')

        self._logger.debug('Choosing Action')
        action = self.current.most_visited()
        self._logger.info(f"Action Chosen {action}")
        self.current, reward, done = self._take_action(self.environment, self.current, action)
        if done:
            self.terminal = True

    def run(self, root):
        """Runs a single iteration of Monte-Carlo tree search."""
        clone_env = self.environment.clone()
        history = []
        history.append(root.id)
        current = root
        done = False
        self._logger.debug("Entering Selection Phase")
        while not done and not current.is_leaf():
            action = self._select(current, clone_env)
            current, reward, done = self._take_action(clone_env, current, action)
            self._logger.debug(f"Selected to State: {current.state}")
            history.append(current.id)

        self._logger.debug("Entering expansion phase")
        if not done:
            action = self._expand(clone_env)
            current, reward, done = self._take_action(clone_env, current, action)
            self._logger.debug(f"Expanded to State: {current.state}")
            history.append(current.id)

        self._logger.debug("Entering simulation phase")
        if not done:
            current, reward, done = self._simulate(current, clone_env)

        self._logger.debug("Entering update phase")
        self._update(reward, history)


    def _select(self, current, env):
        """Selects an action and returns the UCB1 selection"""
        actions = env.actions
        self._logger.debug(all([current.edges.get(action) for action in actions]))

        remaining_actions = [action for action in actions if action not in list(current.edges)]
        if not remaining_actions:
            self._logger.debug("Calling ucb1 selection")
            action = self.policy_manager.selection(current)
        else:
            action = np.random.choice(remaining_actions)
        return action


    def _expand(self, env):
        action = self.policy_manager.rollout(env)
        return action


    def _simulate(self, current, env):
        done = False
        depth = 0
        while not done:
            action = self.policy_manager.rollout(env)
            current, reward, done = self._take_action(env, current, action, add_edge=False)
            self._logger.debug(f"Simulated to state \n{current.state}")
            depth += 1
        self._logger.debug(f"Simulated Depth Reached: {depth}")

        # Make sure that the reward is counted for the player at the expansion node
        # TODO: This is sort of gross. Devise a better way.
        if depth % 2 == 1 and reward == 1:
            reward = 0

        elif reward == 0: # Draw
            # Flip a coin to decide outcome. This should average out to a 50%
            # win rate on expected draw
            reward = np.random.randint(2)

        return current, reward, done


    def _update(self, reward, history):
        win_next_node = reward == 1
        for node_id in reversed(history):
            node = self.nodes[node_id]
            node.visit_count += 1
            if self.adversarial:
                if win_next_node:
                    self._logger.debug(f'Updating win count on state {node.state}')
                    node.win_count += 1
                # Every other node belongs to the same player
                win_next_node = not win_next_node


    def _take_action(self, env, current, action, add_edge=True):
        """Takes the action in the environment and records

        Returns: next_node, reward, done
        """
        observation, reward, done = env.step(action)
        next_node = self._get_node(observation)
        if add_edge and not current.edges.get(action):
            current.edges[action] = self.nodes[next_node.id]

        return next_node, reward, done


    def _get_node(self, state):
        """Grabs the appropriate node for the given state or generates a new one"""
        unique_id = xxhash.xxh64(state).digest()
        if unique_id in self.nodes:
            return self.nodes[unique_id]
        else:
            node = MCTNode(unique_id, state)
            self.nodes[unique_id] = deepcopy(node)
            return node
