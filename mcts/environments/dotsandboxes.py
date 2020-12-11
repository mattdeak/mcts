"""
Created on Sat Apr  8 21:52:43 2017
@author: matthew
"""

import numpy as np
from collections import defaultdict
from numpy import random
from copy import deepcopy


class DotsAndBoxes:
    """The main environment for a dots and boxes game"""

    def __init__(self, size=4):
        """
        Initializes the environment
        :param size: the size (in cells, not dots) of the environment
        """

        self.size = size

        # State is dot-to-dot cells with 5 channels (up,down,left,right, none)
        self.state = np.zeros([size, size, 5])
        self.player = 1
        self.n_players = 2
        self.score = defaultdict(int)
        self.action_space = size * (size + 1) * 2
        self.actions = None
        self.terminal = False
        self.reward_dictionary = {"win": 1, "loss": -1, "draw": 0}
        self.captured_cells = defaultdict(list)
        self.SIDES = {"N": 0, "S": 1, "E": 2, "W": 3}
        self.reset()

    def switch_turn(self):
        """Switches the turn"""
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def end_game(self):
        """Returns final rewards"""
        self.terminal = True
        if self.score[1] == self.score[2]:
            return self.reward_dictionary["draw"]
        elif self.score[self.player] == max(self.score.values()):
            self.winner = self.player
            return self.reward_dictionary["win"]
        else:
            self.winner = 2 if self.player == 1 else 2
            return self.reward_dictionary["loss"]

    def step(self, action):
        """Takes an action and changes the game state."""
        # If an action is invalid and the environment is set to quit
        # on an invalid action, set the reward to loss and return
        # the terminal state. Otherwise, simply return the state without
        # switching turns, to give the player another chance.
        if not self.is_valid_action(action):
            raise ValueError("Invalid Action: {}".format(action))

        else:
            last_state = np.copy(self.state)

            # Add a wall where the action dictates
            self.build_wall(action)

            # Determine the score of the action
            scored = self.score_action(action)
            # Remove the action from the list of valid actions
            self.actions.remove(action)

            # Payout rewards
            reward = 0
            if self.actions == []:
                reward = self.end_game()
                done = True

            else:
                done = False
                if not scored:
                    self.switch_turn()

            return self.state, reward, done

    def score_action(self, action):
        """
        Updates the score based on the action last taken.
        :param action: The integer representing the wall to be built
        :return: True if action scored, false otherwise
        """
        states = self.convert_to_state(action)
        scored = False
        for state_index in states:
            cell_row, cell_column = state_index[:2]
            if np.sum(self.state[cell_row, cell_column]) == 4:
                self.captured_cells[self.player].append([cell_row, cell_column])
                self.score[self.player] += 1
                scored = True

        return scored

    def reset(self):
        self.actions = [i for i in range(self.size * (self.size + 1) * 2)]
        self.score = defaultdict(int)
        self.state = np.zeros([self.size, self.size, 5])
        self.state[:, :, 4] = np.ones([self.size, self.size])
        self.winner = None
        self.player = 1
        self.terminal = False

    def build_wall(self, action):
        """Builds a wall in the game state"""
        states = self.convert_to_state(action)
        for state_index in states:
            row, column, side = state_index
            self.state[row, column, side] = 1
            self.state[row, column, 4] = 0

    def convert_to_wall(self, state):
        """Converts a specific game state array to the wall it represents"""
        row, column, side = state
        if side == self.SIDES["N"]:
            return row * self.size + column
        elif side == self.SIDES["S"]:
            return (row + 1) * self.size + column
        elif side == self.SIDES["E"]:
            return (self.size * (self.size + 1)) + row * self.size + column
        elif side == self.SIDES["W"]:
            return (self.size * (self.size + 1)) + row * self.size + (column + 1)
        else:
            raise ValueError(
                "Can't convert state to wall. 3rd Dimension must be range [0-3]. Value given: {}".format(
                    side
                )
            )

    def convert_to_state(self, wall_number):
        """Converts a wall number to the specific game states that it represents"""
        cells = []
        # If this is true, the wall is on the N-S
        if wall_number < ((self.size) * (self.size + 1)):
            cell_column = wall_number % self.size
            cell_row = wall_number // self.size
            if cell_row != 0:
                cells.append([cell_row - 1, cell_column, self.SIDES["S"]])
            if cell_row != self.size:
                cells.append([cell_row, cell_column, self.SIDES["N"]])
        # Otherwise the wall is E-W
        else:
            cell_row = (wall_number - (self.size * (self.size + 1))) // (self.size + 1)
            cell_column = wall_number % (self.size + 1)
            if cell_column != 0:
                cells.append([cell_row, cell_column - 1, self.SIDES["E"]])

            if cell_column != self.size:
                cells.append([cell_row, cell_column, self.SIDES["W"]])

        return cells

    def is_valid_action(self, action):
        """Ensures that an action is valid"""
        states = self.convert_to_state(action)
        for state_index in states:
            r, c, s = state_index
            if self.state[r, c, s] == 1:
                return False
        return True

    def board(self):
        print(self)

    def clone(self):
        return deepcopy(self)

    def __str__(self):
        """Provides a console output of the current state"""
        string = ""
        final_row = []
        for row in range(self.size):
            column_walls = []

            for column in range(self.size):
                string += "{:<1}".format(".")
                n, s, e, w, _ = self.state[row, column]
                n_string = ""
                if n == 1:
                    n_string = "----"
                else:
                    n_string = " "
                string += "{:<4}".format(n_string)

                if w == 1:
                    column_walls.append("{:<1}".format("|"))
                else:
                    column_walls.append("{:<1}".format(" "))

                if e == 1 and column == self.size - 1:
                    column_walls.append("{:<1}".format("|"))

                if column == self.size - 1:
                    string += "{:<1}".format(".")

                if s == 1 and row == self.size - 1:
                    final_row.append("{:<4}".format("____"))
                elif s == 0 and row == self.size - 1:
                    final_row.append("{:<4}".format(" "))

            string += "\n"
            for wall in column_walls:
                string += wall
                string += "{:<4}".format(" ")
            string += "\n"

        for wall in final_row:
            string += "{:<1}".format(".")
            string += wall

        string += ".\n"

        string += "Player 1 Score is {}\nPlayer 2 Score is {}\n".format(
            self.score[1], self.score[2]
        )

        return string

    def print_state(self, state):
        """Provides a console output of the current state"""
        string = ""
        final_row = []
        for row in range(self.size):
            column_walls = []

            for column in range(self.size):
                string += "{:<1}".format(".")
                n, s, e, w, _ = state[row, column]
                n_string = ""
                if n == 1:
                    n_string = "----"
                else:
                    n_string = " "
                string += "{:<4}".format(n_string)

                if w == 1:
                    column_walls.append("{:<1}".format("|"))
                else:
                    column_walls.append("{:<1}".format(" "))

                if e == 1 and column == self.size - 1:
                    column_walls.append("{:<1}".format("|"))

                if column == self.size - 1:
                    string += "{:<1}".format(".")

                if s == 1 and row == self.size - 1:
                    final_row.append("{:<4}".format("____"))
                elif s == 0 and row == self.size - 1:
                    final_row.append("{:<4}".format(" "))

            string += "\n"
            for wall in column_walls:
                string += wall
                string += "{:<4}".format(" ")
            string += "\n"

        for wall in final_row:
            string += "{:<1}".format(".")
            string += wall

        string += ".\n"

        string += "\n{} Score is {}\n{} Score is {}\n".format(
            self._player1,
            self._player2,
            self.score[self._player1],
            self.score[self._player2],
        )

        return string
