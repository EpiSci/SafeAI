# Copyright (c) 2018 Episys Science, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.estimator import model_fn
from tensorflow.train import get_global_step

from safeai.utils.distribution import kld_with_uniform
from safeai.models.joint_confident.default_submodels import load_default_submodel


def confident_classifier(features, labels, mode, params):
    """ confident_classifier: defines joint_confident model_fn
        Expected params: {
            'image': [feature_x], : numeric_column object with shape
            'noise': [feature_z],
            'classes': num_classes
            'discriminator': None,    : Uninstantiated keras model function. 
            'generator': None,        : To see how to define your own submodel, check `default_submodels.py` 
            'classifier': None,       : Omit the value, or give `None` to each submodel key to use default submodel(dcgan, vgg16)
            'learning_rate': 0.001,
            'alpha': 2.0,
            'beta': 1.0,}
    """

    feature_x = params['image']  # feature_column.numeric_column type object
    feature_z = params['noise']  # that has shape property which excludes batch size (W,H,C)

    input_x = tf.feature_column.input_layer(features, feature_x)
    input_z = tf.feature_column.input_layer(features, feature_z)

    # Input_x shape has already been flattened to (-1, W*H*C) at this point,
    # so reshape again for convolutions
    inferable_shape_x = [-1] + list(feature_x.shape)
    input_x = tf.reshape(input_x, inferable_shape_x)

    # Argument that corresponds to default submodel defined in `default_submodels.py`,
    # in case of the submodel_fn not provided through param
    default_model_args = {
        'classifier': [input_x, params['classes']],
        'generator': [input_z, input_x],
        'discriminator': [input_x]
    }

    submodels = {}
    for submodel_id in ['classifier', 'generator', 'discriminator']:
        if submodel_id not in params or not params[submodel_id]:
            args = default_model_args[submodel_id]
            submodel_fn = load_default_submodel(submodel_id)
            submodels[submodel_id] = submodel_fn(*args)
        else:
            fn, args = params[submodel_id]
            assert isinstance(args, dict)
            submodel = fn(**args)  # Instantiate the model with passed dict arguments
            if not callable(submodel):
                raise ValueError("Model should be callable.")
            submodels[submodel_id] = submodel


    if mode not in [model_fn.ModeKeys.TRAIN,
                    model_fn.ModeKeys.EVAL,
                    model_fn.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: {}'.format(mode))

    classifier = submodels['classifier']
    discriminator = submodels['discriminator']
    generator = submodels['generator']

    classifier_logits = classifier(input_x)
    predicted_classes = tf.argmax(classifier_logits, axis=1)
    confident_score = tf.nn.softmax(classifier_logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': predicted_classes,
            'probabilities': confident_score,
            'logits': classifier_logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    d_score_real = discriminator(input_x)
    generated_fake_image = generator(input_z)
    d_score_fake = discriminator(generated_fake_image)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_score_real,
            labels=tf.ones_like(d_score_real)))

    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_score_fake,
            labels=tf.zeros_like(d_score_fake)))

    discriminator_loss = d_loss_real + d_loss_fake

    g_loss_from_d_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_score_fake,
            labels=tf.ones_like(d_score_fake))
    )

    classifier_logits_on_fake = classifier(generated_fake_image)
    classifier_prob_on_fake = tf.nn.softmax(classifier_logits_on_fake)

    # KLD(softmax(classifier(fake)) || Uniform)
    kld = tf.reduce_mean(kld_with_uniform(classifier_prob_on_fake))

    generator_loss = params['alpha'] * g_loss_from_d_fake + (params['beta'] * kld)

    nll_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=classifier_logits)

    classifier_loss = nll_loss + (params['beta'] * kld)

    # Separate variables to apply gradient only to its own subgraph
    classifier_variables = classifier.trainable_variables
    discriminator_variables = discriminator.trainable_variables
    generator_variables = generator.trainable_variables


    # Define three training operations
    lr = params['learning_rate']
    optim_d = tf.train.AdamOptimizer(lr)
    train_op_d = optim_d.minimize(discriminator_loss,
                                  global_step=get_global_step(),
                                  var_list=discriminator_variables)

    optim_g = tf.train.AdamOptimizer(lr)
    train_op_g = optim_g.minimize(generator_loss,
                                  global_step=get_global_step(),
                                  var_list=generator_variables)

    optim_c = tf.train.AdamOptimizer(lr)
    train_op_c = optim_c.minimize(classifier_loss,
                                  global_step=get_global_step(),
                                  var_list=classifier_variables)

    grouped_ops = tf.group([train_op_d, train_op_g, train_op_c])
    grouped_loss = classifier_loss + generator_loss + discriminator_loss

    # Define accuracy, metrics
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes)

    metrics = {'accuracy': accuracy}

    # Tensorboard visualization
    # (specify model_dir at Estimator initialization)
    tf.summary.image('img_real', tf.reshape(input_x[:2], inferable_shape_x))
    tf.summary.image('img_fake', tf.reshape(generated_fake_image[:2],
                                            inferable_shape_x))
    tf.summary.scalar('d_loss_real', d_loss_real)
    tf.summary.scalar('d_loss_fake', d_loss_fake)
    tf.summary.scalar('KLD(softmax(Classifier(fake)) || Uniform)', kld)
    tf.summary.scalar('classifier_loss', classifier_loss)
    tf.summary.scalar('generator_loss', generator_loss)
    tf.summary.scalar('discriminator_loss', discriminator_loss)

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
