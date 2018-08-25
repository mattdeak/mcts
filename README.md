# mcts
A library for easily building Monte Carlo Tree Search models.

### Installation
Simply pip install this git repo with the following code
```
pip install git+https://github.com/mattdeak/mcts.git
```

### Usage Instructions
Below is example code of how to configure a Monte Carlo Tree Search model with this library.
```
from mcts.environments import TicTacToe
from mcts.mcts import MCTS

config = {
    'action' : 'most-visited',
    'selection' : 'ucb1',
    'selection_kwargs' : {'C' : 1.14},
    'expansion' : 'vanilla',
    'simulation' : 'random-to-end',
    'update' : 'vanilla'
}

# Initialize Environment
tictactoe = TicTacToe()

# Initialize MCTS
ai = MCTS(tictactoe, calculation_time=5)

# Build the MCTS with the config dictionary
ai.build(config)
```

Each model is composed of five different policies.
* The `action` policy determines the method by which the model will choose an action _after_ the monte carlo tree search has been completed.
* The `selection` policy determines the method by which the model chooses actions in the _selection_ phase of the tree search.
* The `expansion` policy determines how the model chooses actions in the expansion phase.
* The `simulation` policy determines how the simulation is run. SiimulateToEnd, for example, will simulate games until termination. 
 Note that the simulation policy also requires a `rollout` policy. This will determine which policy the model uses to select actions in the simulation phase.
 * The `update` policy handles updating the nodes at the end of a single monte-carlo tree search.
 
 Some policies have optional keyword arguments, which can be specified using the `_kwargs` specification. This is shown above to set
 
 To _act_ in an environment, simply call the `act()` method. This will run an MCTS for the number of seconds provided in `calculation_time`, and then choose the best action according to the `action` policy. E.g
 ```
 ai.act()
 ```
 
 #### Available Policy Choices
 To view available choices for each policy, simply inspect the output of the following code:
 ```
 mcts.SUPPORTED_POLICY_TYPES
 ```
 
### Required Environment API
The MCTS is designed to be flexible so that it can be easily plugged into suitable game environments. The required api is inspired heavily by the OpenAI `gym` API, but which a few modifications:
* A `state` attribute, which describes the current state. It's recommended to use a numpy array.
* An `actions` attribute. This must return a list of _valid actions_ in the current game state.
* A `player` attribute which returns the current player. This can be any unique identifier.
* An `n_players` attribute (int) which returns the number of players in the game.
* A `clone` method which clones the environment.
* A `winner` attribute which provides the identity of the winner of the game.
* A `terminal` attribute (bool) which flags whether or not the game is in a terminal state.
* A `step` method which takes an `action`. This will perform the action in the environment and return:
  * observation - the state of the next 
  * reward - The reward for taking that action
  * done - A boolean which is `true` if the action led to a terminal state and `false` otherwise.

Future versions will eliminate some of these requirements.

For neural-network integrated MCTS, plese refer to [this jupyter notebook tutorial](https://github.com/mattdeak/mcts/blob/master/tutorials/Neural%20Configuration%20Tutorial.ipynb).
