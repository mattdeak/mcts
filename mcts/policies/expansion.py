from ..base.policy import NodeTrackingPolicy
from ..utils import softmax
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
        """Performs expansion on node.
        
        Arguments:
            node {mcts.tree.Node} -- The node to expand.
            actions {numpy.array[int]} -- Available actions at this state
        """
        node.expanded = True
        policy_logits, value = self.model.predict_from_node(node)

        # The policy output is in logit form.
        # We need to softmax it to turn it into priors.
        priors = softmax(policy_logits[0])
        valid_priors = priors[actions]

        node.set_edges(actions, priors=valid_priors)
        node.set_value(value[0][0])
        
        