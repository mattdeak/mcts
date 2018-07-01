class SynchronousTrainer:
    """Continuously trains a neural net from a replay table component."""
    def __init__(self, replay, default_batch_size=32):
        self.replay = replay
        self.default_batch_size = 32

    def set_model(self, model):
        self.model = model

    def train_batches(self, n_batches, batch_size=None):
        """Trains the model for a number of batches"""
        if batch_size == None:
            batch_size = self.default_batch_size
    
        for i in range(n_batches):
            input_states, action_values, rewards = self.replay.get_batch(batch_size)
            self.model.fit(input_states, {'policy_head':action_values, 'value_head':rewards})