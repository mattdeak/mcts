import xxhash
import logwood
import numpy as np
import datetime
from sortedcontainers.sorteddict import SortedDict
from copy import deepcopy
from tree.gametree import GameTree



class MCTS:
    def __init__(self, environment, action_policy, selection_policy, rollout_policy,
                update_policy, calculation_time):
        self.tree = GameTree()
       
        self.environment = environment
        
        
        self.choose = action_policy
        self.select = selection_policy
        self.rollout = rollout_policy
        self.simulation_policy = simulation_policy
        self.update_policy = update_policy

        self.calculation_time = calculation_time
        

    @property
    def calculation_time(self):
        return self._calculation_time

    @calculation_time.setter
    def calculation_time(self, seconds):
        self._calculation_time = datetime.timedelta(seconds=seconds)

    def reset():
        self.terminal = False
        self.game_history = []
        self.tree.reset()
    
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
            action = self.selection_policy(current, env_clone)
            self.history.append(current.id)

        # Expansion Phase
        if not done:
            self.tree.expand(env_clone.state, env_clone.valid_actions)
            action = self.rollout_policy(current, env_clone)

        # Simulation Phase
        if not done:
            current, reward, done = self.simulation_policy(current, env_clone)

        # Update Phase
        self.update_policy(current, env_clone, winner, reward, history)