from ..base.policy import NodeTrackingPolicy
import xxhash

class VanillaExpansion(NodeTrackingPolicy):
    """Expands the leaf node by adding possible actions
    as edges to the node."""
    def __call__(self, node, actions):

        node.expanded = True
        node.set_edges(actions)

class NNExpansion(NodeTrackingPolicy):
    """Expands a node using priors based on a neural net.
    
    As of now, only neural nets with both a value-output and a policy-output will be supported."""
    def __init__(self, model):
        self.model = model

    def __call__(self, node, actions):
        policy, value = self.model.predict(node.state)

        # We only care about the priors for
        # valid actions from the current state
        valid_policies = policy[actions]

        node.set_edges(actions, priors)
        node.set_value(value)
        
        