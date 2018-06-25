class Vanilla:

    def add_tree(self, tree):
        self.tree = tree

    def __call__(self, environment, reward, history):
        """Updates the nodes in the node tree.
        
        Increments visit count and win count if the node is winning"""
        winner = environment.winner

        # TODO: Handle for non 2-player games
        if winner == None:
            winner = np.random.randint(clone_env.n_players)

            reward = np.random.randint(2)

        for node_id in history:
            node = self.tree.get_by_id(node_id)
            node.n += 1
            if node.player == node.winner:
                node.w += 1
