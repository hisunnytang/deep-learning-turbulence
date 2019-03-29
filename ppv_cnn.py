#!/usr/bin/python3
# coding: utf-8

# In[10]:


"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import numpy as np
import tensorflow as tf
from os import listdir
import gc
import os
import glob
import h5py
from ppv_cnn_input import *
from ppv_cnn_train import USE_TWO_LAYER, USE_MEAN_SQUARED, PREDICT_BOTH

def two_layers_cnn( input_layer ):
    """two layer cnn model
    Args:
      input_layer: Tensor [batch_size, NXCHANNELS, NVCHANNELS, 1]
    Return:
      flat_layer: Tensor [batch_size, -1]
    """
    # Convolutional Layer #1
    # Computes 8 features using a 4x4 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape:  [batch_size, NXCHANNELS, NVCHANNELS, 1]
    # Output Tensor Shape: [batch_size, NXCHANNELS, NVCHANNELS, 8]

    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 8,
        kernel_size = [4,4],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 8 filter and stride of 2
    # Input Tensor Shape:  [batch_size, 64, 64]
    # Output Tensor Shape: [batch_size, 32, 32, 8]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(4,4), strides=4)

    # Convolutional Layer #2
    # Computes 16 features using a 4x4 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape:  [batch_size, 32, 32, 8 ]
    # Output Tensor Shape: [batch_size, 32, 32, 16]
    conv2 = tf.layers.conv2d(
      inputs  = pool1,
      filters = 16,
      kernel_size = [4,4],
      padding     ="same",
      activation  =tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2 filter and stride of 2
    # Input Tensor Shape:  [batch_size, 32, 32, 16]
    # Output Tensor Shape: [batch_size, 16, 16, 16]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(4,4), strides=4)
    
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape:  [batch_size, 4, 4, 16]
    # Output Tensor Shape: [batch_size, 4x4x16  ]
    pool2_flat = tf.reshape(pool2, [-1, 4*4*16  ])

    return pool2_flat

def three_layers_cnn( input_layer ):
    """three layer cnn model
    Args:
      input_layer: Tensor [batch_size, NXCHANNELS, NVCHANNELS, 1]
    Return:
      flat_layer: Tensor [batch_size, -1]
    """
    # Convolutional Layer #1
    # Computes 8 features using a 4x4 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape:  [batch_size, NXCHANNELS, NVCHANNELS, 1]
    # Output Tensor Shape: [batch_size, NXCHANNELS, NVCHANNELS, 8]

    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 8,
        kernel_size = [4,4],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 8 filter and stride of 2
    # Input Tensor Shape:  [batch_size, 64, 64]
    # Output Tensor Shape: [batch_size, 32, 32, 8]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2,2), strides=2)

    # Convolutional Layer #2
    # Computes 16 features using a 4x4 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape:  [batch_size, 32, 32, 8 ]
    # Output Tensor Shape: [batch_size, 32, 32, 16]
    conv2 = tf.layers.conv2d(
      inputs  = pool1,
      filters = 16,
      kernel_size = [4,4],
      padding     ="same",
      activation  =tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2 filter and stride of 2
    # Input Tensor Shape:  [batch_size, 32, 32, 16]
    # Output Tensor Shape: [batch_size, 16, 16, 16]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2,2), strides=2)
    
    # Convolutional Layer #3
    # Computes 16 features using a 4x4 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape:  [batch_size, 16, 16, 16 ]
    # Output Tensor Shape: [batch_size, 16, 16, 32 ]
    conv3 = tf.layers.conv2d(
      inputs  = pool2,
      filters = 32,
      kernel_size = [4,4],
      padding     ="same",
      activation  =tf.nn.relu)
    
    # Pooling Layer #2
    # Second max pooling layer with a 2 filter and stride of 2
    # Input Tensor Shape:  [batch_size, 16, 16, 32]
    # Output Tensor Shape: [batch_size,  4,  4, 32]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(4,4), strides=4)
  

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape:  [batch_size, 4, 4, 32]
    # Output Tensor Shape: [batch_size, 4x4x32  ]
    pool3_flat = tf.reshape(pool3, [-1, 4*4*32  ])

    return pool3_flat

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # input layer should be of shape [:, NXCHANNELS, NVCHANNELS, 1]
    # NVCHANNELS: number of velocity bins
    
    NVCHANNELS=64
    NXCHANNELS=64
    
    input_layer = tf.reshape(features["x"], [-1,NXCHANNELS, NVCHANNELS, 1])
    
    # Intermediate Layers are specified in different function 
    if USE_TWO_LAYER:
        flat_layer = two_layers_cnn(input_layer) 
    else:
        flat_layer = three_layers_cnn(input_layer)
 
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 4x4x32]
    # Output Tensor Shape: [batch_size, 32]
    dense = tf.layers.dense(inputs=flat_layer, units=32, activation=tf.nn.relu)
    
    # Add dropout operation; 0.7 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
    
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 1]
    if PREDICT_BOTH:
        n_outputs = 2
    else:
        n_outputs = 1
    logits = tf.layers.dense(inputs=dropout, units=n_outputs)
    
    
    if PREDICT_BOTH:
        beta_pred =  logits[:,0]
        gamma_pred = logits[:,1]
        predictions = { "beta_pred": beta_pred, "gamma_pred": gamma_pred }
    else:
        beta_pred = logits[:,0]
        predictions = {"beta_pred": beta_pred}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)   
    # Simple Mean Squared
    if USE_MEAN_SQUARED:
        if PREDICT_BOTH: 
            loss_gamma = tf.losses.mean_squared_error( labels['gamma'] , predictions['gamma_pred'] )
        else:
            loss_gamma = 0.0
        loss_beta  = tf.losses.mean_squared_error( labels['beta'] , predictions['beta_pred']  )
        loss       = loss_beta + loss_gamma
    else:
        # Compute Weighted Loss
        # Fractional difference from its true value
        ones = tf.ones( tf.shape( labels['beta'] ) , dtype=tf.float64 )
        if PREDICT_BOTH:
            inverse_gamma = tf.div( ones, labels['gamma'] )
            loss_gamma = tf.losses.mean_squared_error( labels['gamma'], predictions['gamma_pred'], weights= inverse_gamma )
        else:
            loss_gamma = 0.0
        inverse_beta = tf.div( ones, labels['beta'] )
        loss_beta = tf.losses.mean_squared_error( labels['beta'], predictions['beta_pred'], weights= inverse_beta )
        loss      = loss_beta + loss_gamma 

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        starter_learning_rate = 1.0e-3
        learning_rate = tf.train.exponential_decay(starter_learning_rate, 
            tf.train.get_global_step(), 1000000, 0.96, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if USE_MEAN_SQUARED:
        if PREDICT_BOTH: 
            gamma_accuracy = tf.metrics.mean_squared_error( labels['gamma'] , predictions['gamma_pred'] )
        beta_accuracy  = tf.metrics.mean_squared_error( labels['beta'] , predictions['beta_pred']  )
    else:
        # Compute Weighted Loss
        # Fractional difference from its true value
        ones = tf.ones( tf.shape( labels['beta'] ) , dtype=tf.float64 )
        if PREDICT_BOTH:
            inverse_gamma = tf.div( ones, labels['gamma'] )
            gamma_accuracy = tf.metrics.mean_squared_error( labels['gamma'], predictions['gamma_pred'], weights= inverse_gamma )
        inverse_beta = tf.div( ones, labels['beta'] )
        beta_accuracy = tf.metrics.mean_squared_error( labels['beta'], predictions['beta_pred'], weights= inverse_beta )


    if PREDICT_BOTH: 
        eval_metric_ops = { "beta_accuracy": beta_accuracy, "gamma_accuracy": gamma_accuracy }
    else:
        eval_metric_ops = { "beta_accuracy": beta_accuracy } 
    print(eval_metric_ops)

    return tf.estimator.EstimatorSpec( mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



