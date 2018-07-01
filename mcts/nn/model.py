import tempfile, shutil
import numpy as np
from keras.models import load_model

class Model:
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

        # Save and restore a model 
        with tempfile.NamedTemporaryFile() as temp:
            self.model.save(temp.name)
            clone = load_model(temp.name)
            
        clone.name = self.name + '_clone'
        clone.set_weights(self.model.get_weights())

        return Model(clone, callbacks=self.callbacks)

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, callbacks=self.callbacks, **kwargs)

    def fit_generator(self, generator, **kwargs):
        return self.model.fit_generator(generator, **kwargs)

    def predict(self, X, **kwargs):
        # If it's a single sample, add a dimension
        
        return self.model.predict(X, **kwargs)

    def predict_from_node(self, node, **kwargs):
        X = np.expand_dims(node.state, axis=0)
        return self.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def __getattr__(self, attr):
        return self.model.__getattribute__(attr)