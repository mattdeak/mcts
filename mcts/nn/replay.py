import numpy as np
from numpy import random
import pickle

def load_replay(path):
    """Loads a replay table from a pickle file.
    
    Arguments:
        path {str} -- The path to the saved replay table.
    
    Returns:
        BasicReplay -- The replay table object
    """
    return pickle.load(open(path, 'rb'))

class BasicReplay:
    """A basic replay table.
    
    Stores state, policy-value and state-value information in numpy arrays.
    Can be used as a generator to randomly select from the replay table."""

    def __init__(self, state_shape, policy_size, capacity=50000):
        self.policy_size = policy_size
        self._capacity = 50000
        self._insertion_index = 0
        
        self._states = np.zeros([self._capacity, *state_shape])
        self._policy_values = np.zeros([self._capacity, policy_size])
        self._state_values = np.zeros(self._capacity)


    def add_data(self, states, policy_values, state_values):
        """Adds data to the replay table.
        
        Arguments:
            states {np.Array or tf.Tensor} -- The state information.
            policy_values {np.Array or tf.Tensor} -- The policy search probabilities.
            state_values {float} -- The reward from a given state.
        """
        n = states.shape[0]

        start = self._insertion_index % self._capacity
        end = start + n
  
        self._states[start:end, ...] = states
        self._policy_values[start:end, ...] = policy_values
        self._state_values[start:end] = state_values
            
        self._insertion_index += n


    @property
    def size(self):
        return min(self._insertion_index, self.capacity)

    def save(self, filepath):
        """Saves the replay table as a pickled object.
        
        Arguments:
            filepath {str} -- The save filepath for the replay table.
        """
        pickle.dump(self, open(filepath, 'wb'))

    def get_batch(self, batch_size):
        """Gets a random batch from the replay table.
        
        Arguments:
            batch_size {int} -- The number of samples in the batch.
        
        Returns:
            List -- A list that contains :
                State information for batch_size samples
                Policy information for batch_size samples
                Value information for batch_size samples
        """
        if self._insertion_index == 0:
            ix = 0
        
        elif self._insertion_index < batch_size:
            ix = random.choice(np.minimum(self._insertion_index, self._capacity), size=self._insertion_index, replace=False)
        else:
            ix = random.choice(np.minimum(self._insertion_index, self._capacity), size=batch_size, replace=False)
        
        return self._states[ix, ...], self._policy_values[ix, ...], self._state_values[ix, ...]
        