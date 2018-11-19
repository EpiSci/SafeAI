from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from safeai.utils import gradual_sequence

def load_default_submodel(submodel_id):
    submodels = {
        'classifier': default_classifier,
        'generator': default_generator,
        'discriminator': default_discriminator
    }
    submodel_fn = submodels[submodel_id]
    return submodel_fn  # Returns one of functions above


def default_classifier(input_tensor, classes):
    """Create default classifier.
        Args:
            input_tensor: (N, W, H, C) or (N, C, W, H) tf.float32 tensor
            classes: number of classes
        Returns:
            model: tf.keras.Model instance
    """
    image_shape = input_tensor.shape[1:] # (N, 32, 32, 3) => (32, 32, 3)
    image_dim_flatten = np.prod(image_shape) # (32, 32, 3) => 3072
    # gradual_sequence() returns a list of integers that gradually
    # decreases/increases. For example, gradual_sequence(784, 10, multiplier=3)
    # returns a list [270, 90, 30]
    units_in_layers = gradual_sequence(image_dim_flatten, classes)

    # Define default model
    inputs = tf.keras.layers.Input(shape=image_shape, name='input')
    net = tf.keras.layers.Flatten()(inputs)
    net = tf.keras.layers.Dense(image_dim_flatten, activation='relu')(net)
    for hidden_units in units_in_layers:
        net = tf.keras.layers.Dense(hidden_units,
                                    activation='relu')(net)
    logits = tf.keras.layers.Dense(classes, activation=None)(net)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

def default_vgg16(input_tensor, classes):
    inputs = tf.keras.layers.Input(input_tensor.shape[1:])
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(2048, activation='relu', name='fc1')(x)
    logits = tf.keras.layers.Dense(classes, activation=None)(x)
    vgg = tf.keras.models.Model(inputs=inputs, outputs=logits)
    return vgg


def default_generator(input_tensor, image_tensor):
    """Default generator network.
        Args:
            noise_input: (N, latent_space_size), tf.float32 tensor
            image_tensor: img tensor to get shape of fake img, tf.float32 tensor
        Returns:
            model: tf.keras.Model instance
    """
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


def default_discriminator(input_tensor):
    """Default discriminator network.
        Args:
            input_tensor: (None, input_image_dim), tf.float32 tensor
        Returns:
            model: tf.keras.Model instance
    """
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
    return model
