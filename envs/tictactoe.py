import numpy as np
from copy import deepcopy

class TicTacToe:
    """A toy tictactoe environment for testing MCTS"""
    def __init__(self):
        self.reset()

    @property
    def actions(self):
        actions = []
        for i in range(self.state.size):
            if self._state[i] == 0:
                actions.append(i)
        return actions

    @property
    def state(self):
        return self._state.reshape([3, 3])

    def step(self, action):
        assert action in self.actions

        if self.player == 1:
            self._state[action] = 1
        else:
            self._state[action] = -1

        win = self._check_win()

        self._rotate_players()

        if win:
            self.terminal = True
            return self.state, 1, True
        elif self._board_full():
            self.terminal = True
            return self.state, 0, True
        return self.state, 0, False

    def reset(self):
        self._state = np.zeros(9)
        self.player = 1
        self.terminal = False

    def _check_win(self):
        # Check rows
        state = self.state
        for row in state:
            if sum(row) == 3 or sum(row) == -3:
                return True

        for column in state.T:
            if sum(column) == 3 or sum(column) == -3:
                return True

        # Check diagonal
        if sum(state.diagonal()) == 3 or sum(state.diagonal()) == -3:
            return True

        if np.fliplr(state).diagonal().sum() == 3 or np.fliplr(state).diagonal().sum() == -3:
            return True
        else:
            return False

    def _board_full(self):
        return not 0 in self._state

    def _rotate_players(self):
        self.player = 1 if self.player == 2 else 2

    def clone(self):
        return deepcopy(self)
