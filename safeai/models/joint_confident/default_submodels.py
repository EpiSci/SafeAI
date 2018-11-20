from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
import numpy as np

from safeai.utils import gradual_sequence

def load_default_submodel(submodel_id):
    submodels = {
        'classifier': tiny_vgg16,
        'generator': dcgan_generator,
        'discriminator': dcgan_discriminator,
    }
    submodel_fn = submodels[submodel_id]
    return submodel_fn  # Returns one of functions above


def default_classifier(input_tensor, classes):
    image_shape = input_tensor.shape[1:]
    image_dim_flatten = np.prod(image_shape)
    units_in_layers = gradual_sequence(image_dim_flatten, classes)

    inputs = tf.keras.layers.Input(shape=image_shape, name='input')
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(image_dim_flatten, activation='relu')(x)

    for hidden_units in units_in_layers:
        x = tf.keras.layers.Dense(hidden_units, activation='relu')(x)

    logits = tf.keras.layers.Dense(classes, activation=None)(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


def tiny_vgg16(input_tensor, classes):

    input_shape = input_tensor.shape[1:]
    with tf.name_scope("classifier"):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten(name='flatten'))
        model.add(Dense(1024, activation='relu', name='fc1'))
        model.add(Dense(classes, activation=None, name='logits'))

    return model


def default_generator(input_tensor, image_tensor):
    image_shape = image_tensor.shape[1:]
    image_dim_flatten = np.prod(image_shape)
    assert len(input_tensor.shape) == 2  # (N, latent_space_size)
    noise_dim = input_tensor.shape[1]
    units_in_layers = gradual_sequence(noise_dim, image_dim_flatten)

    inputs = tf.keras.layers.Input(shape=(noise_dim,))
    net = tf.keras.layers.Dense(noise_dim, activation='relu')(inputs)

    for hidden_units in units_in_layers:
        net = tf.keras.layers.Dense(hidden_units, activation='relu')(net)
    image_flatten = tf.keras.layers.Dense(image_dim_flatten,
                                          activation=None)(net)

    output_layer = tf.keras.layers.Reshape(image_shape)(image_flatten)

    model = tf.keras.Model(inputs=inputs, outputs=output_layer)
    return model

# Todo: Need more general method for determining node numbers 
# in each layer Tue 20 Nov 2018 02:00:50 AM KST
def dcgan_generator(input_tensor, image_tensor):

    image_shape = image_tensor.shape[1:] # (W, H, C)
    if len(image_shape) == 3:
        output_channel = int(image_shape[2])
    else:
        output_channel = 1

    nodes_factor = image_shape[0] // 4 # 7 for 28x28, 8 for 32x32

    assert len(input_tensor.shape) == 2  # (N, latent_space_size)
    noise_dim = input_tensor.shape[1]

    with tf.name_scope("generator"):
        model = Sequential()
        model.add(Dense(2048, input_shape=(noise_dim,)))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(256 * nodes_factor * nodes_factor))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())

        model.add(Reshape((nodes_factor, nodes_factor, 256),
                          input_shape=(256*nodes_factor*nodes_factor,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(128, (2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(32, (4, 4), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(16, (4, 4), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(output_channel, (5, 5), padding='same'))
        model.add(Activation('tanh'))

    return model

def default_discriminator(input_tensor):
    image_shape = input_tensor.shape[1:]
    image_dim_flatten = np.prod(image_shape)
    units_in_layers = gradual_sequence(image_dim_flatten, 1)

    inputs = tf.keras.layers.Input(shape=(image_shape))
    net = tf.keras.layers.Flatten()(inputs)
    net = tf.keras.layers.Dense(image_dim_flatten, activation='relu')(net)
    for hidden_units in units_in_layers:
        net = tf.keras.layers.Dense(hidden_units, activation='relu')(net)
    logit = tf.keras.layers.Dense(1, activation=None)(net)
    model = tf.keras.Model(inputs=inputs, outputs=logit)
    r# in each layer Tue 20 Nov 2018 02:00:50 AM KST

def dcgan_discriminator(input_tensor):
    image_shape = input_tensor.shape[1:]

    with tf.name_scope("discriminator"):
        model = Sequential()
        model.add(Conv2D(64, (5, 5),
                        padding='same',
                        input_shape=image_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1))
        model.add(LeakyReLU(0.2))
    return model
