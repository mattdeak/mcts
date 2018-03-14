from .utils import  build_rescnn

class BaseNN:
    """The base neural network class for MCTS integration.

    By default supports initialization with Keras models, but for other frameworks
    just override the necessary methods.

    Models must have an M-dimensional policy head, where M
    is the dimensionality of the action space.

    Models may also have a value head. This
    """
    def __init__(self, model, callbacks=[]):
        self.model = model
        self.callbacks = callbacks

    def clone(self):
        clone = keras.utils.clone_model(self)
        clone.set_weights(self.model.get_weights())
        return clone

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, callbacks=self.callbacks, **kwargs=**kwargs)

    def predict(self, X, kwargs):
        return self.model.predict(X, **kwargs)

    def save(self, filepath):
        """Saves the model architecture and weights"""

    def load(self, filepath):
        """Loads a model from filepath"""

class ZeroNN(BaseNN):

    def __init__(self, env, residual_layers=40, filters=256, kernel_size=3):
        model = zeronet(env.state.shape,
                             env.action_space.shape,
                             filters, kernel_size,
                             residual_layers=residual_layers)
        super(ZeroNN, self).__init__(model)
