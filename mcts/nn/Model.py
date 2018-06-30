import tempfile, shutil
from keras.models import model_from_json

class BaseNN:
    """The base neural network class for MCTS integration.

    By default supports initialization with Keras models, but for other frameworks
    just override the necessary methods.

    Models must have an M-dimensional policy head, where M
    is the dimensionality of the action space.

    Models may also have a value head. This
    """
    def __init__(self, model, callbacks=[], **kwargs):
        self.model = model
        self.callbacks = callbacks
        self.kwargs = kwargs

    def clone(self):
        """Makes a compiled clone of the current model."""

        # Create temporary files for the model information
        temp_model = tempfile.TemporaryFile(mode='w+t')
        temp_weights = tempfile.TemporaryFile(mode='w+t')

        model_json = self.model.to_json()
        temp.write(model_json)

        self.model.save_weights(temp_weights)


        clone.set_weights(self.model.get_weights())

        return BaseNN(clone, callbacks=self.callbacks)

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, callbacks=self.callbacks, **kwargs)

    def fit_generator(self, generator, **kwargs):
        return self.model.fit_generator(generator, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def __getattr__(self, attr):
        return self.model.__getattribute__(attr)