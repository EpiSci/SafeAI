from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.estimator import model_fn
from tensorflow.train import get_global_step

from safeai.utils.distribution import kl_divergence_with_uniform
from safeai.models.joint_confident.default_submodels import load_default_submodel


def confident_classifier(features, labels, mode, params):
    """ confident_classifier: defines joint_confident model_fn
        Expected params: {
            'image': [image_feature],
            'noise': [noise_feature],
            'classes': num_classes
            'discriminator': None,  <-
            'generator': None,      <- instantiated keras model,
            'classifier': None,     <- or chain of callable layers
            'learning_rate': 0.01,
            'beta': 1.0,
        }
    """
    image_feature = params['image']
    noise_feature = params['noise']

    image_input_layer = tf.feature_column.input_layer(features,
                                                      image_feature)
    noise_input_layer = tf.feature_column.input_layer(features,
                                                      noise_feature)

    default_model_args = {
        'classifier': [image_input_layer, params['classes']],
        'generator': [noise_input_layer, image_input_layer],
        'discriminator': [image_input_layer]
    }

    submodels = {}
    for submodel_id in ['classifier', 'generator', 'discriminator']:
        if not params[submodel_id] or submodel_id not in params:
            tf.logging.info("Fetching default {}".format(submodel_id))
            args = default_model_args[submodel_id]
            submodel_fn = load_default_submodel(submodel_id)
            submodels[submodel_id] = submodel_fn(*args)
        else:
            if not callable(params[submodel_id]):
                raise ValueError("Model should be callable.")
            submodels[submodel_id] = params[submodel_id]


    if mode not in [model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                    model_fn.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: {}'.format(mode))

    # Combine three independent submodels
    classifier = submodels['classifier']
    discriminator = submodels['discriminator']
    generator = submodels['generator']

    logits = classifier(image_input_layer)
    predicted_classes = tf.argmax(logits, axis=1)
    confident_score = tf.nn.softmax(logits)

    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': predicted_classes,
            'probabilities': confident_score,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Todo: Take loss from params Mon Nov  5 20:14:35 2018
    # Discriminator loss

    # Put real image to discriminator
    d_score_real = discriminator(image_input_layer)

    # Put fake image to discriminator
    generated_fake_image = generator(noise_input_layer)

    d_score_fake = discriminator(generated_fake_image)

    # Delete these soon
    #tf.summary.image('real', tf.reshape(image_input_layer[:10], [-1, 32, 32, 3]))
    #tf.summary.image('fake', tf.reshape(generated_fake_image[:10], [-1, 32, 32, 3]))

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_score_real,
            labels=tf.ones_like(d_score_real))
    )

    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_score_fake,
            labels=tf.zeros_like(d_score_fake))
    )

    # Discriminator loss
    discriminator_loss = d_loss_real + d_loss_fake


    g_loss_from_d = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_score_fake,
            labels=tf.ones_like(d_score_fake))
    )

    logits_fake = classifier(generated_fake_image)

    confident_score_fake = tf.nn.softmax(logits_fake)
    classifier_uniform_kld_fake =\
        tf.reduce_mean(kl_divergence_with_uniform(confident_score_fake))

    # Generator loss
    generator_loss = g_loss_from_d +\
        (params['beta'] * classifier_uniform_kld_fake)

    # Classifier loss
    nll_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)
    classifier_loss = nll_loss + (params['beta'] * classifier_uniform_kld_fake)

    # Separate variables to applying gradient only to subgraph
    classifier_variables = classifier.trainable_variables
    discriminator_variables = discriminator.trainable_variables
    generator_variables = generator.trainable_variables

    # Define three training operations
    lr = params['learning_rate']
    optimizer_discriminator = tf.train.AdamOptimizer(lr)
    train_discriminator_op =\
        optimizer_discriminator.minimize(discriminator_loss,
                                         global_step=get_global_step(),
                                         var_list=discriminator_variables)

    optimizer_generator = tf.train.AdamOptimizer(lr)
    train_generator_op =\
        optimizer_generator.minimize(generator_loss,
                                     global_step=get_global_step(),
                                     var_list=generator_variables)

    optimizer_classifier = tf.train.AdamOptimizer(lr)
    train_classifier_op =\
        optimizer_classifier.minimize(classifier_loss,
                                      global_step=get_global_step(),
                                      var_list=classifier_variables)

    grouped_ops = tf.group([train_discriminator_op,
                            train_generator_op,
                            train_classifier_op])

    tf.summary.scalar('classifier_loss', classifier_loss)
    tf.summary.scalar('generator_loss', generator_loss)
    tf.summary.scalar('discriminator_loss', discriminator_loss)

    grouped_loss = classifier_loss + generator_loss + discriminator_loss

    # Define accuracy, metrics
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes)

    metrics = {'accuracy': accuracy}

    # Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=grouped_loss, eval_metric_ops=metrics)

    # Train: Alternative learning
    if mode == tf.estimator.ModeKeys.TRAIN:
        est_spec = tf.estimator.EstimatorSpec(mode,
                                              loss=grouped_loss,
                                              train_op=grouped_ops)
        return est_spec

    raise ValueError("Invalid estimator mode: reached the end of the function")
