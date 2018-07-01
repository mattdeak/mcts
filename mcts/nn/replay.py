import numpy as np

class BasicReplay:

    def __init__(self, state_shape, policy_size, capacity=50000, batch_size=8):
        self.batch_size = 8
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
        
        
    def __next__(self):
        if self._insertion_index == 0:
            ix = 0
        
        elif self._insertion_index < self.batch_size:
            ix = random.choice(np.minimum(self._insertion_index, self._capacity), size=self._insertion_index, replace=False)
        else:
            ix = random.choice(np.minimum(self._insertion_index, self._capacity), size=self.batch_size, replace=False)
        
        return self._states[ix, ...], [self._action_values[ix, ...], self._state_values[ix, ...]]
        