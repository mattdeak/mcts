from ..base.policy import BasePolicy
import xxhash

class VanillaExpansion(BasePolicy):
    """Exp"""
    def add_tree(self, tree):
        self.tree = tree

    def __call__(self, node, actions):
        node.expanded = True

        node.set_edges(actions)
        
        