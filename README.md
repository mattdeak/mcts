# mcts
A library for easily building Monte Carlo Tree Search models.

### Instructions
Below is example code of how to configure a Monte Carlo Tree Search model with this library.
```
# Import the relevant policies
from mcts.environments import TicTacToe
from mcts.policies.action import MostVisited
from mcts.policies.rollout import RandomUnvisited, RandomChoice
from mcts.policies.selection import UCB1
from mcts.policies.simulation import SimulateToEnd
from mcts.policies.update import Vanilla
from mcts.mcts import MCTS

# Initialize Environment
tictactoe = TicTacToe()

action = MostVisited()
selection = UCB1()
rollout = RandomUnvisited()
sim = SimulateToEnd(RandomChoice())
update = Vanilla()
mcts = MCTS(tictactoe, action, selection, rollout, sim, update, calculation_time=5)
```

Each model is composed of five different policies.
* The `action` policy determines the method by which the model will choose an action _after_ the monte carlo tree search has been completed.
* The `selection` policy determines the method by which the model chooses actions in the _selection_ phase of the tree search.
* The `rollout` policy determines how the model chooses actions in the expansion phase.
* The `simulation` policy determines how the simulation is run. SiimulateToEnd, for example, will simulate games until termination. 
 Note that the simulation policy also requires a `rollout` policy. This will determine which policy the model uses to select actions in the simulation phase.
 * The `update` policy handles updating the nodes at the end of a single monte-carlo tree search.
 
 To _act_ in an environment, simply call the `act()` method. This will run an MCTS for the number of seconds provided in `calculation_time`, and then choose the best action according to the `action` policy. E.g
 ```
 mcts.act()
 ```
 
### Required Environment API
The MCTS is designed to be flexible so that it can be easily plugged into suitable game environments. Any environment must have the following attributes:
* A `state` attribute, which describes the current state. It's recommended to use a numpy array here.
* An `actions` attribute. This must return a list of _valid actions_ in the current game state.
* A `player` attribute which returns the current player. This can be any unique identifier.
* An `n_players` attribute which returns the number of players in the game.
* A `clone` method which clones the environment.
* A `winner` attribute which provides the identity of the winner of the game.
* A `terminal` attribute which flags whether or not the game is in a terminal state.

Currently only policies for a basic MCTS model are supported, but more policies are in development (along with neural network integration).
