import numpy as np
import tensorflow as tf

def kl_divergence_with_uniform(target_distribution):
    # Expects (examples, classes) as shape
    num_classes = tf.cast(target_distribution.shape[1], tf.float32)
    uniform_distribution = tf.divide(tf.ones_like(target_distribution), num_classes)
    x = tf.distributions.Categorical(probs=target_distribution)
    y = tf.distributions.Categorical(probs=uniform_distribution)
    return tf.distributions.kl_divergence(x, y) * num_classes  # scaling factor

