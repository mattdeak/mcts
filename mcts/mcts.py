import xxhash
import logwood
import numpy as np
import datetime
from sortedcontainers.sorteddict import SortedDict
from copy import deepcopy
from .tree.gametree import GameTree


class MCTS:

    supported_policy_types = ['action','selection','expansion'
                             ,'simulation' ,'update','expansion_rollout']

    def __init__(self, environment, calculation_time=5):
        self.tree = GameTree()
       
        self.environment = environment
        self.calculation_time = calculation_time
        self.configured = False

        self.reset()

    def build(self, policy_dict):
        for key, policy in policy_dict.items():
            if key not in self.supported_policy_types:
                raise ValueError(f"{key} is not a supported policy type.")
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

        if self.terminal:
            raise ValueError("Game environment is terminal. Cannot take action.")
        begin = datetime.datetime.utcnow()

        # Run MCTS for the calculation window
        games_played = 0

        while (datetime.datetime.utcnow() - begin < self._calculation_time):
            self.run(current)
            games_played += 1

        # Act in the environment
        action = self.choose(current)
        current, reward, done = self._step(current, action, self.environment)


    def run(self, root):
        assert self.configured, "MCTS must be configured before running."

        env_clone = self.environment.clone()
        history = []
        current = root
        done = False

        # Selection Phase: Use selection policy to traverse
        # game tree until a leaf node (unexpanded) is reached.
        while not done and current.expanded:
            action = self.select(current, env_clone)
            history.append([current.id, action])
            current, reward, done = self._step(current, action, 
                                    env_clone)

        # Expansion Phase
        if not done:
            self.expand(current, env_clone.actions)

            # Perform another rollout if applicable
            if self.expansion_rollout:
                action = self.expansion_rollout(current, env_clone)
                history.append([current.id, action])
                current, reward, done = self._step(current, action, 
                                        env_clone)

        # Simulation Phase if Applicable
        if not done and self.simulate:
            _, reward, done = self.simulate(current, env_clone)

        # Update Phase
        self.update(env_clone, reward, history)

    def _step(self, current, action, environment):
        """Takes a step in the environment"""
        observation, reward, done = environment.step(action)
        player = environment.player

        next_node = self.tree.evaluate(current.id, action, observation, player=player)
        return next_node, reward, done