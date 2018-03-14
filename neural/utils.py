import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Flatten
from keras.layers import Input, add, Activation
from keras.layers import regularizers
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import keras.backend as K

def convolutional_block(x, filters=256, kernel_size=[3,3], c_reg=0.0001):
    x = Conv2D(filters= filters,
             kernel_size=kernel_size,
             padding='same',
             activation='linear',
             kernel_regularizer=regularizers.l2(c_reg))(x)
    x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)

    return x

def residual_block(input_layer, filters=256, kernel_size=[3,3], c_reg=0.0001, name=None):
    x = convolutional_block(input_layer, filters, kernel_size)

    # Add convolution
    x = Conv2D(filters= filters,
             kernel_size=kernel_size,
             padding='same',
             activation='linear',
             kernel_regularizer=regularizers.l2(c_reg)
             )(x)

    x = BatchNormalization(axis=1)(x)

    # Concatenate input layer with
    x = add([input_layer, x])
    x = LeakyReLU(name=name)(x)

    return (x)

def policy_head(x, output_shape, filters=2, kernel_size=[1, 1], c_reg=0.0001):
    x = convolutional_block(x, filters, kernel_size, c_reg)

    x = Flatten()(x)
    x = Dense(output_shape,
              activation='linear',
              kernel_regularizer=regularizers.l2(c_reg),
              name='policy_head')(x)
    return x


def value_head(x, dense_shape=4, filters=1, kernel_size=[1, 1], c_reg=0.0001):
    x = convolutional_block(x, filters, kernel_size, c_reg)
    x = Flatten()(x)
    x = Dense(
        dense_shape,
        activation='linear',
        kernel_regularizer=regularizers.l2(c_reg)
        )(x)
    x = LeakyReLU()(x)
    x = Dense(
        1,
        activation='tanh',
        kernel_regularizer=regularizers.l2(c_reg),
        name='value_head'
        )(x)

    return x

def crossentropy_with_logits(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

def load_zeronet(input_shape, output_shape, residual_layers=40, lr=0.0001, c_reg=0.0001,
                 momentum=0.9):
    # First layer is just convolutional
    input_layer = Input(shape=input_shape, name='Input')
    x = convolutional_block(input_layer)

    for i in range(residual_layers):
        x = residual_block(x, name=f'residual{i}')

    # Output Layers
    policy = policy_head(x, output_shape)
    value = value_head(x)


    model = Model(inputs=input_layer, outputs=[policy, value])

    optimizer = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False)

    model.compile(
        loss={
            'value_head':'mean_squared_error',
            'policy_head':crossentropy_with_logits
            },
        loss_weights={'value_head':0.5, 'policy_head':0.5},
        optimizer=optimizer
        )

    return model

def zeronet_lr_schedule(epoch):
    if epoch < 400:
        return 0.01
    elif epoch < 600:
        return 0.001

    # else
    return 0.0001
