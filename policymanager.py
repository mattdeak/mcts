from abc import ABC, abstractmethod
import asyncio
from collections import deque
import logwood
from utils import ucb1, puct
from numpy.random import choice
from scipy.stats import dirichlet
import threading
from queue import Queue
from simulator.simulator import Simulator
from sortedcontainers import SortedDict
from utils import zero_temperature_choice
import atexit
import sys
import numpy as np

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
    def action_choice(self, node):
        """This policy determines the action to be taken after MCTS has finished."""

    @abstractmethod
    def rollout(self, node):
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

    def action_choice(self, node):
        return node.most_visited_child()

    def rollout(self, node):
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
    def __init__(self, model, environment, checkpoint=1000, validation_games=400,
                    update_threshold=0.55, memory_size=10, alpha=0.03,
                    epsilon=0.25, C=1.41, optimize=True,
                    temperature_schedule={0:1e-2}, batch_size=4, training=False):
        super(NNPolicyManager, self).__init__()
        self._nn_logger = logwood.get_logger(self.__class__.__name__ + '|Network')

        self._next_model = model
        self.model = model.clone()


        self._initialize_models()
        self._input_dims = model.input.shape.ndims

        self.alpha = alpha
        self.C = C

        self._training = training



        self._state_shape = environment.state.shape
        self._n_policies = environment.action_space

        self.simulator = Simulator(environment)
        self.checkpoint_interval = checkpoint
        self.checkpoint = 1
        self._training_step = 0
        self.epsilon = epsilon
        self.validation_games = validation_games
        self.update_threshold = update_threshold

        self.replay_states = np.zeros([memory_size, *[dim for dim in self._state_shape]])
        self.replay_policies = np.zeros([memory_size, self._n_policies])
        self.replay_rewards = np.zeros(memory_size)

        self.replay_maxlen = memory_size
        self._replay_index = 0
        self.min_train_capacity = memory_size // 2

        self._train_thread = threading.Thread(target=self._train_process)
        self.batch_size = batch_size

        # For threading
        self.replay_ready = threading.Event()
        self._optimize_event = threading.Event()
        self._exit_thread = threading.Event()

        if optimize:
            self._optimize_event.set()
            self._train_thread.start()

        self.temperature_schedule = SortedDict(temperature_schedule)
        atexit.register(self._cleanup)


    def start_optimization(self):
        self._exit_thread.clear()
        self._optimize_event.set()

    def halt_optimization(self):
        self._optimize_event.clear()

    def stop_optimization(self):
        self.optimize_event.clear()
        self._exit_thread.set()

    def _initialize_models(self):
        self.model._make_predict_function()

        self._next_model._make_predict_function()
        self._next_model._make_test_function()
        self._next_model._make_train_function()

    def _cleanup(self):
        self._exit_thread.set()
        self._train_thread.join()

    def action_choice(self, node, move_number=None):
        if move_number:
            temp = [i for j, i in self.tempurature_schedule.items() if j < 5][-1]
            return zero_temperature_choice(node, temp)
        else:
            return zero_temperature_choice(node)

    def rollout(self, node, env):
        # We'll just use the selection policy here.
        return self.selection(node, env, root=False)

    def selection(self, node, env, root=False):
        """Uses PUCT to determine selected action."""
        actions = env.actions
        probabilities = self.nn_policy_from_node(node)
        # Add a little noise to the root node to add exploration
        if root:
            probabilities = ((1-self.epsilon)*probabilities
                              + self.epsilon*np.random([self.alpha]*len(probabilities)))

        pi = puct(node, probabilities, self.C)
        # Get the best valid action
        return actions[pi[actions].argmax()]


    def nn_policy_from_node(self, node):
        # Single prediction
        X = np.expand_dims(node.state, axis=0)
        # If the model wants convolutional layers
        # But we only have M X N, make it M X N X 1
        if X.ndim == 3 and self._input_dims == 4:
            X = X[:, :, :, np.newaxis]

        return self.model.predict(X)[0][0]


    def update(self, node, reward):
        self._logger.info("Adding state to replay table")
        i = self._replay_index % self.min_train_capacity
        # We add the node to the replay table
        edge_values = [node[i].value for i in range(self._n_policies)]


        self.replay_states[i] = node.state
        self.replay_policies[i] = edge_values
        self.replay_rewards[i] = reward

        self._replay_index += 1
        # Trigger datapoint added event
        if not self.replay_ready.is_set() and self._replay_index > self.min_train_capacity:
            self._logger.info("Replay Table Ready")
            self.replay_ready.set()

    def _train_process(self):
        # Shortcuts
        self._nn_logger.info("Starting Training Thread")
        min_capacity = self.min_train_capacity / 2
        batch_size = self.batch_size
        while not self._exit_thread.is_set():
            try:
                while self._optimize_event.is_set():
                    self._nn_logger.info("Waiting for more data")
                    self.replay_ready.wait()
                    while (self._optimize_event.is_set()):

                        idx = idx = np.random.randint(
                            min(self._replay_index, self.replay_maxlen),
                            size=batch_size
                        )

                        X = self.replay_states[idx]
                        y1 = self.replay_policies[idx]
                        y2 = self.replay_rewards[idx]

                        history = self._next_model.fit(X, [y1, y2], initial_epoch=self._training_step)
                        self._nn_logger.info(history)

                        self._training_step += 1
                        if self._training_step == self.checkpoint:
                            self.checkpoint += self.checkpoint_interval
                            self._next_model.save(f'checkpoint {self.checkpoint}')
                            eval_thread = threading.Thread(target=self.evaluation)
                            eval_thread.start()
                            eval_thread.join()
                    self._optimize_event.wait()
            except Exception as e:
                self._nn_logger.error(f"Train Process Crashed: {e}")
                sys.exit(0)


    def evaluation(self):
        """Runs a tournament between two MCTS agents. If the next neural network
        functions better as a rollout policy than the current one, we update
        the policy so that it's using the new one."""
        self._logger.info("Running Evaluation at timestep {self._training_step}")

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
                optimize=False))

        next_mcts = MCTS(
            env,
            NNPolicyManager(
                model=self._next_model,
                alpha=self.alpha,
                C=self.C,
                temperature=1e-99,
                optimize=False))

        results = self.simulator.simulate([current_agent, next_agent], n=validation_games)
        # TODO: If next agent won more than win_threshold percent, switch models
        if results[2]['wins']/self.validation_games > self.update_threshold:
            self.info("New model wins! Updating threshold")
            self.model = self._next_model.clone()

class ZeroPolicyManager(NNPolicyManager):
    """Policy Manager for AlphaGo Zero style play.

    Uses a residual convolutional network for policy and value evaluations."""
    def __init__(self, environment, residual_layers=40, filters=256, kernel_size=3, **kwargs):
        input_shape = environment.state.shape
        output_shape = environment.action_space

        model = ZeroNN(
            input_shape,
            output_shape,
            residual_layers=residual_layers,
            filters=filters,
            kernel_size=kernel_size)

        super(ZeroPolicyManager, self).__init__(model, environment, **kwargs)
