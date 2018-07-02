class SynchronousTrainer:
    """Continuously trains a neural net from a replay table component."""
    def __init__(self, replay, callbacks=[], default_batch_size=32):
        self.replay = replay
        self.default_batch_size = 32
        self.callbacks = callbacks

    def set_model(self, model):
        self.model = model

    def train_batches(self, n_batches, batch_size=None):
        """Trains the model for a number of batches"""
        if batch_size == None:
            batch_size = self.default_batch_size
        
        generator = self._make_generator(batch_size)
        validation_data = next(generator)

        # Since this is a reinforcement learning problem, validation data doesn't really mean anything. 
        # However, we need to specify it for some callbacks (e.g tensorboard), 
        # so we'll just reuse the data generator.
        self.model.fit_generator(generator, 
                                 steps_per_epoch=1,
                                 epochs=n_batches,
                                 callbacks=self.callbacks,
                                 validation_data=validation_data
                                 )
    

    def _make_generator(self, batch_size):
        def generator(batch_size):
            while True:
                input_states, action_values, rewards = self.replay.get_batch(batch_size)
                X, y = input_states, {'policy_head':action_values, 'value_head':rewards}
                yield X, y
        return generator(batch_size)