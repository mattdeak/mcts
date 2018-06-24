from sortedcontainers import SortedList()

class Edge:
    """The edge class for the game tree.
    
    This class stores the visit count, q-value and priors of edges in the tree.
    When evaluated, an edge will point to a Node."""
    def __init__(self, action, prior=0):
        self.action = action
        self.n = 0
        self.q = 0
        self.p = prior

    def evaluate(self, node):
        self.evaluated = True
        self.node = node

class Node:
    """The Node class for the game tree.
    
    This class stores the state information in the game tree. Each node will contain a list
    of edges upon being expanded."""
    def __init__(self, state, player=None):
        self.id = xxhash.xxh64(state_id).digest() # Used to identify state
        self.state = state
        self.player = player
        self.expanded = False

    def __hash__(self):
        return self.id

    def expand(self, actions, priors=[]):
        if priors:
            self.edges = [Edge(actions[i], priors[i]) for i in range(len(actions))]
        else:
            self.edges = [Edge(action) for action in actions)]
        self.expanded = True
        
    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

class GameTree:
    """Contains the game tree for the MCTS"""
    def __init__(self):
        self.nodes = {}
        
    def evaluate(self, parent_id, action, state):
        """Adds a node to the node tree"""
        node = Node(state)
        self.nodes[parent_id][action].evaluate(node)

    def get_by_id(self, node_id):
        """Retrieves a node by the node ID"""
        return self.nodes[node_id]

    def get_by_state(self, state):
        state_id = xxhash.xxh64(state).digest()

        node = self.nodes.get(state_id)
        if node:
            return node

        else:
            node = Node(state)
            self.nodes[state_id] = node
            return node

    def reset(self):
        self.nodes = {}

    def expand(self, state_id, actions):
        # Flag the expanded node
        self.nodes[state_id].expand(actions)


