import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Flatten
from keras.layers import Input, add, Activation
from keras.layers import regularizers
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import keras.backend as K
import numpy as np


def add_convolutional_block(model, filters=256, kernel_size=[3, 3], c_reg=0.0001):
    """Adds a convolutional block to a model."""
    model = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation="linear",
        use_bias=False,
        kernel_regularizer=regularizers.l2(c_reg),
    )(model)
    model = BatchNormalization()(model)
    model = LeakyReLU()(model)

    return model


def add_residual_block(
    input_layer, filters=256, kernel_size=[3, 3], c_reg=0.0001, name=None
):
    """Adds a residual block to a model."""
    model = add_convolutional_block(input_layer, filters, kernel_size)

    # Add convolution
    model = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation="linear",
        use_bias=False,
        kernel_regularizer=regularizers.l2(c_reg),
    )(model)

    model = BatchNormalization()(model)

    # Concatenate input layer with
    model = add([input_layer, model])
    model = LeakyReLU(name=name)(model)

    return model


def add_policy_head(model, output_shape, filters=2, kernel_size=[1, 1], c_reg=0.0001):
    """Adds a policy head to a model."""
    model = add_convolutional_block(model, filters, kernel_size, c_reg)

    model = Flatten()(model)
    model = Dense(
        output_shape,
        activation="linear",
        kernel_regularizer=regularizers.l2(c_reg),
        use_bias=False,
        name="policy_head",
    )(model)
    return model


def add_value_head(model, dense_shape=4, filters=1, kernel_size=[1, 1], c_reg=0.0001):
    """Adds a value head to a model."""
    model = add_convolutional_block(model, filters, kernel_size, c_reg)
    model = Flatten()(model)
    model = Dense(
        dense_shape,
        activation="linear",
        kernel_regularizer=regularizers.l2(c_reg),
        use_bias=False,
    )(model)
    model = LeakyReLU()(model)
    model = Dense(
        1,
        activation="tanh",
        kernel_regularizer=regularizers.l2(c_reg),
        use_bias=False,
        name="value_head",
    )(model)

    return model


def add_policy_value_heads(
    model,
    input_layer,
    policy_output,
    policy_filters=2,
    policy_kernel=[1, 1],
    policy_reg=0.0001,
    value_dense_shape=4,
    value_filters=1,
    value_kernel=[1, 1],
    value_reg=0.0001,
):
    """Adds a double head to a model"""
    policy = add_policy_head(
        model,
        policy_output,
        filters=policy_filters,
        kernel_size=policy_kernel,
        c_reg=policy_reg,
    )

    value = add_value_head(
        model,
        dense_shape=value_dense_shape,
        filters=value_filters,
        kernel_size=value_kernel,
        c_reg=value_reg,
    )

    return Model(inputs=input_layer, outputs=[policy, value])


def crossentropy_with_logits(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def load_zeronet(
    input_shape, output_shape, residual_layers=40, lr=0.0001, c_reg=0.0001, momentum=0.9
):
    """Creates a neural net with an architecture similar to that of Alphago Zero"""
    # If the dimension is M x N, make it M x N x 1
    if len(input_shape) == 2:
        input_shape = np.append(input_shape, 1)
    # First layer is just convolutional
    input_layer = Input(shape=input_shape, name="Input")
    x = add_convolutional_block(input_layer)

    for i in range(residual_layers):
        x = add_residual_block(x, name="residual{}".format(i))

    # Output Layers
    policy = add_policy_head(x, output_shape)
    value = add_value_head(x)

    model = Model(inputs=input_layer, outputs=[policy, value])

    optimizer = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False)

    losses = {
        "policy_head": crossentropy_with_logits,
        "value_head": "mean_squared_error",
    }

    loss_weights = {"value_head": 0.5, "policy_head": 0.5}

    model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)

    return model


def zeronet_lr_schedule(epoch):
    if epoch < 400:
        return 0.01
    elif epoch < 600:
        return 0.001

    # else
    return 0.0001
