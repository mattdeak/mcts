from sortedcontainers import SortedList
import xxhash

class Edge:
    """The edge class for the game tree.
    
    This class stores the visit count, q-value and priors of edges in the tree.
    When evaluated, an edge will point to a Node."""
    def __init__(self, action, prior=0):
        self.action = action
        self.n = 0
        self.w = 0
        self.p = prior
        self.evaluated = False

    def evaluate(self, node):
        self.evaluated = True
        self.node = node

    @property
    def q(self):
        if self.n == 0:
            return 0

        return self.w / self.n

class Node:
    """The Node class for the game tree.
    
    This class stores the state information in the game tree. Each node will contain a list
    of edges upon being expanded."""
    def __init__(self, state, player=None):
        self.id = xxhash.xxh64(state).digest() # Used to identify state
        self.state = state
        self.player = player
        self.expanded = False

    def set_edges(self, actions, priors=[]):
        """Sets the edges of the node"""
        if priors == []:
            self.edges = {action : Edge(action) for action in actions}
        else:
            self.edges = {actions[i] : Edge(actions[i], prior=priors[i]) for i in range(len(actions))}

    def set_value(self, value):
        """Sets the value of the node.
        Not all implementations of MCTS require that nodes have this attribute. Whether or not
        To use a value will be determined by the expansion policy."""
        self.value = value

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __getitem__(self, action):
        return self.edges[action]

class GameTree:
    """Contains the game tree for the MCTS"""
    def __init__(self):
        self.nodes = {}
        
    def evaluate(self, parent_id, action, state, player=None):
        """Adds a node to the node tree if 
        the node is not already present
        
        @returns node: Node added to tree"""
        node = self.get_by_state(state, player=player)

        # Evaluate the state-action pair given by
        # parent_id, action if this pair has not already
        # been evaluated.
        if not self.nodes[parent_id][action].evaluated:
            self.nodes[parent_id][action].evaluate(node)

        return node

    def get_by_id(self, node_id):
        """Retrieves a node by the node ID"""
        return self.nodes[node_id]

    def get_by_state(self, state, player=None):
        """Retrieves a node by the associated state.
        
        If a node associated to that state does not exist,
        a node is made and put in the game tree."""
        state_id = xxhash.xxh64(state).digest()

        node = self.nodes.get(state_id)
        if node:
            return node

        else:
            node = Node(state, player=player)
            self.nodes[state_id] = node
            return node

    def reset(self):
        self.nodes = {}


