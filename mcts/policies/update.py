from ..base.policy import BasePolicy
import numpy as np

class Vanilla(BasePolicy):

    def add_tree(self, tree):
        self.tree = tree

    def __call__(self, environment, reward, history):
        """Updates the nodes in the node tree.
        
        Increments visit count and win count if the node is winning"""
        
        winner = environment.winner
        self._logger.debug(f"Updating winner: {winner}")

        # TODO: Handle for non 2-player games
        if winner == None:
            winner = np.random.randint(environment.n_players)
            reward = np.random.randint(2)

        for node_id, action in history:

            node = self.tree.get_by_id(node_id)
            node[action].n += 1
            if node.player == environment.winner:
                node[action].w += 1
