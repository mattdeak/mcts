import tempfile, shutil
import numpy as np
import logwood
from keras.models import load_model
from keras.callbacks import TensorBoard
import tensorflow as tf
from copy import deepcopy

class Model:
    n_models = 0
    """The base neural network class for MCTS integration.

    By default supports initialization with Keras models, but for other frameworks
    just override the necessary methods.

    Models must have an M-dimensional policy head, where M
    is the dimensionality of the action space.

    As of version Models may also have a value head ( . This
    """
    def __init__(self, model, **kwargs):
        name = kwargs.get("name")
        
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__ + str(self.n_models)

        self._logger = logwood.get_logger(self.name)
        self.model = model

        # Allows the model to be multithreaded
        self.model._make_predict_function()
        self._graph = tf.get_default_graph()

        self.kwargs = kwargs

    def clone(self):
        """Makes a compiled clone of the current model."""
        # Save and restore a model 
        with tempfile.NamedTemporaryFile() as temp:
            self.model.save(temp.name)
            clone = load_model(temp.name)
            
        clone.name = self.name + '_clone'

        return Model(clone)

    def predict_from_node(self, node, **kwargs):
        """Runs a prediction from an mcts.Node object directly
        
        Arguments:
            node {mcts.Node} -- The node on which the model is meant to predict
        
        Returns:
            numpy.array -- Predictions from the state information contained in the node
        """
        X = np.expand_dims(node.state, axis=0)
        with self._graph.as_default():
            return self.model.predict(X)

    def predict(self, X):
        """Runs a prediction using this models graph as the default graph.
        
        Arguments:
            X {numpy.array} -- A state input matrix
        
        Returns:
            numpy.array -- Predictions from the model
        """
        with self._graph.as_default():
            return self.model.predict(X)

    def __getattr__(self, attr):
        return self.model.__getattribute__(attr)