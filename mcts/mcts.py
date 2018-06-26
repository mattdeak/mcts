import xxhash
import logwood
import numpy as np
import datetime
from sortedcontainers.sorteddict import SortedDict
from copy import deepcopy
from .tree.gametree import GameTree


class MCTS:
    def __init__(self, environment, action_policy, selection_policy, rollout_policy,
                simulation_policy, update_policy, calculation_time):
        self.tree = GameTree()
       
        self.environment = environment
        
        self.choose = action_policy
        self.select = selection_policy
        self.rollout = rollout_policy
        self.simulate = simulation_policy
        self.update = update_policy
        self.update.add_tree(self.tree)

        self.calculation_time = calculation_time

        self.reset()
        

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
            self.tree.expand(env_clone.state, env_clone.actions)
            action = self.rollout(current, env_clone)
            history.append([current.id, action])
            current, reward, done = self._step(current, action, 
                                    env_clone)

        # Simulation Phase
        if not done:
            _, reward, done = self.simulate(current, env_clone)

        # Update Phase
        self.update(env_clone, reward, history)

    def _step(self, current, action, environment):
        """Takes a step in the environment"""
        observation, reward, done = environment.step(action)
        player = environment.player

        next_node = self.tree.evaluate(current.id, action, observation, player=player)
        return next_node, reward, done