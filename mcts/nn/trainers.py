class SynchronousTrainer:
    """Continuously trains a neural net from a replay table component."""
    def __init__(self, model, replay, limit=None):
        self.model = model
        self.replay = replay
        self.limit = limit

    def train_batches(n_batches):
        """Trains the model on"""
        for i in n_batches:
            input_states, policy_true, value_true= next(replay)
            self.model.fit(input_states, [policy_true, value_true])