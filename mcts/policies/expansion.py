from ..base.policy import BasePolicy
import xxhash

class VanillaExpansion(BasePolicy):
    """Expands the leaf node by adding possible actions
    as edges to the node."""
    def add_tree(self, tree):
        self.tree = tree

    def __call__(self, node, actions):

        node.expanded = True
        node.set_edges(actions)
        
        