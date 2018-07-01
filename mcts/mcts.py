import xxhash
import logwood
import numpy as np
import datetime
from sortedcontainers.sorteddict import SortedDict
from copy import deepcopy
from .tree.gametree import GameTree
from . import SUPPORTED_POLICY_TYPES
from .builder import ConfigBuilder


class MCTS:

    def __init__(self, environment, calculation_time=5, terminal_callback = None, name=None):
        self.tree = GameTree()

        # Configure logger
        if name == None:
            name = self.__class__.__name__

        self._logger = logwood.get_logger(name)
        self.environment = environment
        self.calculation_time = calculation_time
        if terminal_callback:
            self._handle_terminal = terminal_callback
            self._handle_terminal.add_tree(self.tree)

        self._builder = ConfigBuilder()
        self.configured = False

        self.reset()

    def build(self, raw_config):
        config = self._builder.build(raw_config)
        self.policies = config.values()
        for key, policy in config.items():
            if key not in SUPPORTED_POLICY_TYPES:
                raise ValueError("{} is not a supported policy type.".format(key))
            elif key == 'action':
                self.choose = policy
            elif key == 'selection':
                self.select = policy
            elif key == 'expansion':
                self.expand = policy
                self.expand.add_tree(self.tree)
            elif key == 'simulation':
                self.simulate = policy
            elif key == 'update':
                self.update = policy
                self.update.add_tree(self.tree)
            elif key == 'expansion_rollout':
                self.expansion_rollout = policy

        # Check that required policies exist
        if not self.update or not self.select or not self.expand:
            raise ValueError("Config dictionary is missing vital policies. selection, expansion and update policies are required.")

        self.configured = True
             

    @property
    def calculation_time(self):
        return self._calculation_time

    @calculation_time.setter
    def calculation_time(self, seconds):
        self._calculation_time = datetime.timedelta(seconds=seconds)

    def reset(self):
        self.terminal = False
        self.game_history = []
        self.tree.reset()
    
    def act(self):
        assert self.configured, "MCTS must be configured before running."

        state = self.environment.state
        player = self.environment.player
        
        current = self.tree.get_by_state(state, player=player)
        self.game_history.append(current.id)

        if self.terminal:
            raise ValueError("Game environment is terminal. Cannot take action.")
        begin = datetime.datetime.utcnow()

        # Information to collect while running search
        games_played = 0
        max_depth = 0

        # Run MCTS for the calculation window
        while (datetime.datetime.utcnow() - begin < self._calculation_time):
            depth = self.run(current)
            games_played += 1
            max_depth = max(depth, max_depth)

        self._logger.info("Searches Run: {} | Max Depth: {}".format(games_played, max_depth))

        # Act in the environment
        action = self.choose(current)
        current, reward, done = self._step(current, action, self.environment)

        if done:
            if hasattr(self, "_handle_terminal"):
                self._handle_terminal(self.game_history, reward, self.environment.winner)

            self.reset()


    def run(self, root):
        """Runs a single MCTS search based on policies provided during `build`.
        
        Arguments:
            root {mcts.tree.Node} -- The root node to run from
        
        Returns:
            depth -- The depth reached up until the first leaf node
        """
        assert self.configured, "MCTS must be configured before running."

        env_clone = self.environment.clone()
        history = []
        current = root
        done = False
        depth = 0
        reward = 0

        # Selection Phase: Use selection policy to traverse
        # game tree until a leaf node (unexpanded) is reached.
        while not done and current.expanded:
            action = self.select(current)
            history.append([current.id, action])
            current, reward, done = self._step(current, action, 
                                    env_clone)
            
            depth += 1

        # Expansion Phase
        if not done:
            self.expand(current, env_clone.actions)
            depth += 1

            # Perform another rollout if applicable
            try:
                action = self.expansion_rollout(current, env_clone)
                history.append([current.id, action])
                current, reward, done = self._step(current, action, 
                                        env_clone)
                depth += 1
            except Exception:
                self._logger.debug("No Expansion Rollout Phase")

        # Simulation Phase if Applicable
        try:
            if not done:
                _, reward, done = self.simulate(current, env_clone)
        except Exception:
            self._logger.debug("No simulation phase")

        # Update Phase
        self.update(env_clone, reward, history)
        return depth

    def set_policy_attribute(self, policy_tuple):
        """Updates an attribute on all policies that have it.
        
        Arguments:
            policy_tuple {tuple} -- Form of (attr_name, value)
        """
        name, value = policy_tuple
        for policy in self.policies:
            if hasattr(policy, name):
                setattr(policy, name, value)

    def set_terminal_callback(self, callback):
        self._handle_terminal = callback
        self._handle_terminal.add_tree(self.tree)
    
    def self_play(self, games=100):
        for i in range(games):
            self.environment.reset()
            self.reset()
            while not self.environment.terminal:
                self.act()

    def _step(self, current, action, environment):
        """Takes a step in the environment"""
        observation, reward, done = environment.step(action)
        player = environment.player

        next_node = self.tree.evaluate(current.id, action, observation, player=player)
        return next_node, reward, done