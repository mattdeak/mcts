from keras.models import clone_model
from keras.callbacks import LearningRateScheduler
from .utils import zeronet_lr_schedule

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
        clone = clone_model(self.model)
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

    def compile(self):
        self.model.compile(
            loss=self.losses,
            optimizer=self.optimizer,
            loss_weights=self.kwargs.get('loss_weights',[0.5, 0.5])
        )

    def save(self, filepath):
        """Saves the model architecture and weights"""

    def load(self, filepath):
        """Loads a model from filepath"""

    def __getattr__(self, attr):
        return self.model.__getattribute__(attr)

class ZeroNN(BaseNN):
    """Implements an AlphaGo Zero style residual network"""
    def __init__(self, input_shape, output_shape, residual_layers=40, filters=256, kernel_size=3):
        model = zeronet(input_shape,
                        output_shape,
                        filters,
                        kernel_size,
                        residual_layers=residual_layers)

        callbacks = [LearningRateScheduler(zeronet_lr_schedule)]

        super(ZeroNN, self).__init__(model, callbacks)
