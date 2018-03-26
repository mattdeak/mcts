import xxhash
import logwood
import numpy as np
from .utils import ucb1
from .policymanagers import DefaultPolicyManager
import datetime
from sortedcontainers.sorteddict import SortedDict
from copy import deepcopy


class MCTNode:
    """Node class for MCTS. Tracks state and win information."""
    def __init__(self, state_id, state, player_turn):
        self.id = state_id # Used to identify state
        self.state = state
        self.player = player_turn
        self.visit_count = 0
        self.win_count = 0
        self.edges = SortedDict()

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        else:
            return self.win_count / self.visit_count

    def most_visited_child(self):
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
        return self.edges.get(action, MCTNode(1, None, None))
        # Return Dummy MCTNode for never visited

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class MCTS:

    def __init__(self, environment, policy_manager=None, C=1.41, calculation_time=5,
                 max_simulations=None):
        self.environment = environment
        self._logger = logwood.get_logger(self.__class__.__name__)
        self._calculation_time = datetime.timedelta(seconds=calculation_time)
        self.C = C

        if max_simulations:
            self.max_simulations = max_simulations
        else:
            self.max_simulations = np.inf

        if policy_manager:
            self.policy_manager = policy_manager
        else:
            self.policy_manager = DefaultPolicyManager(C=C)

        self.terminal = False
        self.game_history = []

        self.reset()

    @property
    def calculation_time(self):
        return self._calculation_time

    @calculation_time.setter
    def calculation_time(self, seconds):
        self._calculation_time = datetime.timedelta(seconds=seconds)

    def reset(self):
        self._logger.info("Resetting MCTS Nodes")
        self.terminal = False
        self.nodes = {}
        self.game_history = []
        self._reset_environment()

    def _reset_environment(self):
        """Resets the game environment"""
        self.environment.reset()
        state = self.environment.state
        player = -1 # Nobody moved into the starting state
        self.current = self._get_node(state, player=player)
        self.game_history.append(self.current.id)
        self.move_number = 1


    def act(self):
        """Takes the next action in the current game environment."""
        state = self.environment.state
        player = self.environment.last_player
        self.current = self._get_node(state, player)

        if self.current.id not in self.game_history:
            self._logger.info(f"Adding ID {self.current.id} to history")
            self.game_history.append(self.current.id)



        if self.terminal:
            raise ValueError("Game environment is terminal. Cannot take action.")
        begin = datetime.datetime.utcnow()

        # Run MCTS for the calculation window
        games_played = 0

        while ((datetime.datetime.utcnow() - begin < self._calculation_time) and
               games_played < self.max_simulations):
            self.run(self.current)
            games_played += 1

        self._logger.info(f'Games Simulated: {games_played}')

        self._logger.debug('Choosing Action')
        action = self.policy_manager.action_choice(self.current)
        self._logger.info(f"Action Chosen {action}")

        self.current, reward, done = self._take_action(self.environment, self.current, action)
        self._logger.info(f"Adding Node {self.current.id} to game dictionary.")
        self.move_number += 1
        winner = self.environment.winner

        if self.current.id not in self.game_history:
            self.game_history.append(self.current.id)


        if done:
            self._handle_termination(reward, winner)


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
            action = self._expand(current, clone_env)
            current, reward, done = self._take_action(clone_env, current, action)
            self._logger.debug(f"Expanded to State: {current.state}")
            history.append(current.id)

        self._logger.debug("Entering simulation phase")
        if not done:
            current, reward, done = self._simulate(current, clone_env)

        # Draws will approximate to a 50% win.
        winner = clone_env.winner
        if winner == None:
            winner = np.random.randint(clone_env.n_players) + 1
            reward = np.random.randint(2)

        self._update(reward, winner, history)


    def _select(self, current, env):
        """Selects an action and returns the UCB1 selection"""
        action = self.policy_manager.selection(current, env)
        return action


    def _expand(self, current, env):
        action = self.policy_manager.rollout(current, env)
        return action


    def _simulate(self, current, env):
        done = False
        depth = 0
        while not done:
            action = self.policy_manager.rollout(current, env)
            current, reward, done = self._take_action(env, current, action, add_edge=False)
            self._logger.debug(f"Simulated to state \n{current.state}")
            depth += 1
        self._logger.debug(f"Simulated Depth Reached: {depth}")

        return current, reward, done


    def _update(self, reward, winner, history):
        for node_id in reversed(history):
            node = self.nodes[node_id]
            node.visit_count += 1
            if node.player == winner:
                self._logger.debug(f'Updating win count on state {node.state}')
                node.win_count += 1


    def _handle_termination(self, reward, winner):
        self.terminal = True
        for node_id in self.game_history:
            node = self.nodes[node_id]
            if self.environment.winner == node.player and reward != 0:
                reward = 1
            elif reward != 0:
                reward = -1
            self.policy_manager.update(node, reward)


    def _take_action(self, env, current, action, add_edge=True):
        """Takes the action in the environment and records

        Returns: next_node, reward, done
        """
        player = env.player
        observation, reward, done = env.step(action)
        next_node = self._get_node(observation, player=player)
        if add_edge and not current.edges.get(action):
            current.edges[action] = self.nodes[next_node.id]

        return next_node, reward, done


    def _get_node(self, state, player=None):
        """Grabs the appropriate node for the given state or generates a new one"""
        unique_id = xxhash.xxh64(state).digest()
        if unique_id in self.nodes:
            return self.nodes[unique_id]
        else:
            if player == None:
                raise ValueError(f"Can't create node for state {state} - no player specified")

            node = MCTNode(unique_id, state, player)
            self.nodes[unique_id] = deepcopy(node)
            return node
