from abc import ABC, abstractmethod
from collections import deque
import logwood
from utils import ucb1, puct
from numpy.random import choice
from scipy.stats import dirichlet
from multiprocessing.pool import Pool
class BasePolicyManager(ABC):
    """
    The base class for the MCTS policy manager.

    By inheriting this class and implementing the policy
    methods, it should be quite simple to build a policy using neural networks
    or other heuristic methods.
    """
    def __init__(self):
        self._logger = logwood.get_logger(self.__class__.__name__)

    @abstractmethod
    def rollout(self, state):
        """This policy determines what action is taken in the expansion and
        simulation phases of the MCTS."""

    @abstractmethod
    def selection(self, node):
        """This policy determines what action is taken during the selection
        phase of the MCTS."""

    @abstractmethod
    def update(self, node, reward):
        """Implementing this policy allows future policies to depend on
        play results. If this is unimportant to your policy, simply
        pass this method."""

class DefaultPolicyManager(BasePolicyManager):
    """
    The default policy manager for MCTS.

    An MCTS object using this policy manager will use random rollout and
    UCB1 selection.
    """
    def __init__(self, ucb1):
        super(DefaultPolicyManager, self).__init__()
        self.C = C

    def rollout(self, env):
        return choice(env.actions)

    def selection(self, node):
        return ucb1(node)

    def update(self, node, reward):
        pass

class NNPolicyManager(BasePolicyManager):
    """A policy manager for neural network rollout policies.
    Default settings are based on AlphaGo Zero.

    Input:
        model (BaseNN Child): The model to use.
        simulator (Simulator): The game playing simulator. Used to evaluate
            neural networks against the current one.
        update_interval (int):  The number of games that are played between
            updating the neural network policy.
        validation_games (int): The number of games that are played when evaluating
            whether to update to the next network.
        update_threshold (float 0 < x < 1): The win rate the new network
            must achieve in the validation games to replace the old network.
        memory_size (int): The number of samples to retain in the replay table.
        alpha (float 0 < x < 1): The Alpha to use in generating dirichlet noise
        """
    def __init__(self, model, simulator=None, checkpoint=1000, validation_games=400,
                    update_threshold=0.55, memory_size=10000, alpha=0.03,
                    C=1.41, batch_size=32, temperature=0.05, training=False):
        super(NNPolicyManager, self).__init__()
        self._nn_logger = logwood.get_logger(self.__class__.__name__ + '|Network')
        self.model = model

        self.memory_size = memory_size
        self.alpha = alpha
        self.C = C
        self.temperature = temperature
        self.training = training

        self._next_model = model.clone()

        if self.training:
            self._next_model = model.clone()
            self.simulator = simulator
            self.checkpoint_interval = checkpoint
            self.checkpoint = 1
            self.validation_games = validation_games
            self.update_threshold = update_threshold
            self.replay_table = deque(maxlen=memory_size)


    def rollout(self, node):
        # We'll just use the selection policy here.
        return self.selection(node, root=False)

    def selection(self, node, root=False):
        """Uses PUCT to determine selected action."""
        probabilities = self.model.predict(node.state)
        # Add a little noise to the root node to add exploration
        if root:
            probabilities = dirichlet(probabilities)

        return puct(node, probabilities, self.C)


    def update(self, node, reward):
        # We add the node to the replay table
        self.replay_table.append([node.state, node.value, reward])


    def _train_process(self):
        for i in range(self.checkpoint):
            X, y1, y2 = np.random.choice(self.replay_table, self.batch_size, replace=False)
            history = self._next_model.fit(X, y)
            self._nn_logger.info(history)

        self.checkpoint += self.checkpoint_interval
        self._next_model.save(f'checkpoint {self.checkpoint}')
        self.simulate()


    def evaluation(self):
        """Runs a tournament between two MCTS agents. If the next neural network
        functions better as a rollout policy than the current one, we update
        the policy so that it's using the new one."""
        if not self.simulator:
            raise ValueError("Cannot run NNPolicyManager in training mode without a simulator")

        env = self.simulator.environment

        current_mcts = MCTS(
            env,
            NNPolicyManager(
                model=self.model,
                alpha=self.alpha,
                C=self.C,
                temperature=1e-99,
                testing=True))

        next_mcts = MCTS(
            env,
            NNPolicyManager(
                model=self._next_model,
                alpha=self.alpha,
                C=self.C,
                temperature=1e-99,
                testing=True))

        results = self.simulator.simulate([current_agent, next_agent], n=validation_games)
        # TODO: If next agent won more than win_threshold percent, switch models
        if results[2]['wins']/self.validation_games > self.update_threshold:
            self.model.set_weights(self._next_model.get_weights())
