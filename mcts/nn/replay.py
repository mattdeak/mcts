import numpy as np
from numpy import random

class BasicReplay:
    """A basic replay table.
    
    Stores state, action-value and state-value information in numpy arrays.
    Can be used as a generator to randomly select from the replay table."""

    def __init__(self, state_shape, policy_size, capacity=50000):
        self.policy_size = policy_size
        self._capacity = 50000
        self._insertion_index = 0
        
        self._states = np.zeros([self._capacity, *state_shape])
        self._action_values = np.zeros([self._capacity, policy_size])
        self._state_values = np.zeros(self._capacity)

    def add_data(self, states, action_values, state_values):
        n = states.shape[0]

        start = self._insertion_index % self._capacity
        end = start + n
  
        self._states[start:end, ...] = states
        self._action_values[start:end, ...] = action_values
        self._state_values[start:end] = state_values
            
        self._insertion_index += n

    @property
    def size(self):
        return min(self._insertion_index, self.capacity)
        
        
    def get_batch(self, batch_size):
        if self._insertion_index == 0:
            ix = 0
        
        elif self._insertion_index < batch_size:
            ix = random.choice(np.minimum(self._insertion_index, self._capacity), size=self._insertion_index, replace=False)
        else:
            ix = random.choice(np.minimum(self._insertion_index, self._capacity), size=batch_size, replace=False)
        
        return self._states[ix, ...], self._action_values[ix, ...], self._state_values[ix, ...]
        