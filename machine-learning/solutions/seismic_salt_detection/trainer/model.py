#!/usr/bin/env python
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines TensorFlow model.

Defines features and classification model.

Typical usage example:

model.create_classifier(config, parameters)
"""

import math

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def forward_key_to_export(estimator):
    """Forwards record key to output during inference.

    Temporary workaround. The key and its value will be extracted from input
    tensors and returned in the prediction dictionary. This is useful to pass
    record key identifiers. Code came from:
    https://towardsdatascience.com/how-to-extend-a-canned-tensorflow-estimator-to-add-more-evaluation-metrics-and-to-pass-through-ddf66cd3047d
    This shouldn't be necessary. (CL/187793590 was filed to update extenders.py
    with this code)

    Args:
        estimator: `Estimator` being modified.

    Returns:
        A modified `Estimator`
    """
    config = estimator.config

    def model_fn2(features, labels, mode):
        estimator_spec = estimator._call_model_fn(
            features, labels, mode, config=config)
        if estimator_spec.export_outputs:
            for ekey in ['predict', 'serving_default']:
                estimator_spec.export_outputs[
                    ekey
                ] = tf.estimator.export.PredictOutput(
                    estimator_spec.predictions)
        return estimator_spec
    return tf.estimator.Estimator(model_fn=model_fn2, config=config)


def create_classifier(config, parameters):
    """Creates a DNN classifier.

    Defines features and builds an 'Estimator' with them.

    Args:
        config: `RunConfig` object to configure the runtime of the `Estimator`.
        parameters: Parameters passed to the job.

    Returns:
        A configured and ready to use `tf.estimator.DNNClassifier`
    """
    # Mean and Standard Deviation Constants for normalization.
    mean = np.float32(parameters.depth_mean)
    std = np.float32(parameters.depth_std)

    # Columns to be used as features.

    depth = tf.feature_column.numeric_column(
        'depth',
        normalizer_fn=(lambda x: (x - mean) / std))

    image = hub.image_embedding_column('image', parameters.tf_hub_module)

    feature_cols = [depth, image]

    def estimator_metrics(labels, predictions):
        """Creates metrics for Estimator.

        Metrics defined here can be used to evaluate the model (on evaluation
        data) and also can be used to maximize or minimize their values during
        hyper-parameter tunning.

        Args:
            labels: Evaluation true labels.
            predictions: Evaluation model predictions.

        Returns:
            A dictionary with the evaluation metrics
        """
        pred_logistic = predictions['logistic']
        pred_class = predictions['class_ids']
        return {
            'accuracy': tf.metrics.accuracy(labels, pred_class),
            'auc': tf.metrics.auc(labels, pred_logistic),
            'auc_pr': tf.metrics.auc(labels, pred_logistic, curve='PR'),
            'precision': tf.metrics.precision(labels, pred_class),
            'recall': tf.metrics.recall(labels, pred_class)}

    layer = parameters.first_layer_size
    lfrac = parameters.layer_reduction_fraction
    nlayers = parameters.number_layers
    h_units = [layer]
    for _ in range(nlayers - 1):
        h_units.append(math.ceil(layer * lfrac))
        layer = h_units[-1]

    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_cols,
        hidden_units=h_units,
        optimizer=tf.train.AdagradOptimizer(
            learning_rate=parameters.learning_rate),
        dropout=parameters.dropout, config=config)
    estimator = tf.contrib.estimator.add_metrics(
        estimator, estimator_metrics)
    estimator = tf.contrib.estimator.forward_features(estimator, 'id')
    estimator = forward_key_to_export(estimator)
    return estimator
    