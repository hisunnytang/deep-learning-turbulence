#!/usr/bin/python3
# coding: utf-8



"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import numpy as np
from os import listdir
import gc
import os
import glob
import h5py
from ppv_cnn import *
from ppv_cnn_input import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

use_two_layer = False

def train_2d_cnn(datadir, model_name):
    # handcrafted training sample

    filenames = glob.glob( os.path.join(datadir, "*h5") )
    filenames.sort(key=os.path.getmtime) 
    # avoid reading data that havent finished generating
    filenames = filenames[:-12] 

    X, Y = get_batch(filenames, N = 128)
    
    # Create the Estimator
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    gpu_options = tf.GPUOptions( allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig(session_config = sess_config)  
    beta_estimator = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir=model_name, config=run_config)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    #     tensors_to_log = {"beta": "softmax_tensor"}
    #     logging_hook = tf.train.LoggingTensorHook(
    #       tensors=tensors_to_log, every_n_iter=50)



    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X},
        y=Y,
        batch_size=32,
        num_epochs=None,
        shuffle=True)

    beta_estimator.train(
        input_fn=train_input_fn,
        steps=5000)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X},
        y=Y,
        num_epochs=1,
        shuffle=False)


    eval_results = beta_estimator.evaluate(input_fn=eval_input_fn)
    print("eval results:", eval_results)
    
    X_valid, Y_valid = get_batch(filenames, N = 64)
    beta_estimator = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir=model_name, config=run_config)
    beta_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_valid},
        y=Y_valid,
        num_epochs=1,
        shuffle=False)
    beta_predict = beta_estimator.predict(input_fn=beta_input_fn)
    i = 0
    
    current_step = beta_estimator.get_variable_value("global_step")
    eval_dict = { "beta_pred": [], "gamma_pred": [], "beta": [], "gamma": [] }
    for bbb in beta_predict:
        eval_dict['beta_pred'].append( bbb['beta_pred'] )
        eval_dict['gamma_pred'].append( bbb['gamma_pred'] )
        eval_dict['beta'].append(  Y_valid["beta"][i] )
        eval_dict['gamma'].append( Y_valid["gamma"][i])
        i += 1

   
    return beta_estimator, eval_dict, current_step

def validate_on_ppv_cube( filename, N = 10000 ):
    X_valid, Y_valid = get_batch(filenames, N = 64)
    beta_estimator = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir=model_name, config=run_config)
    beta_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_valid},
        y=Y_valid,
        num_epochs=1,
        shuffle=False)
    beta_predict = beta_estimator.predict(input_fn=beta_input_fn)
    i = 0
    
    current_step = beta_estimator.get_variable_value("global_step")
    eval_dict = { "beta_pred": [], "gamma_pred": [], "beta": [], "gamma": [] }
    for bbb in beta_predict:
        eval_dict['beta_pred'].append( bbb['beta_pred'] )
        eval_dict['gamma_pred'].append( bbb['gamma_pred'] )
        eval_dict['beta'].append(  Y_valid["beta"][i] )
        eval_dict['gamma'].append( Y_valid["gamma"][i])
        i += 1

    return beta_estimator, eval_dict, current_step

datadir = "/home/kwoksun2/deep-learning-turbulence/generateIC/ppvdata"
model_name = "cnn_3layers_noisy_smoothing"
model_performance_dir = model_name + "_performance"




if __name__ == "__main__":
    if not os.path.exists( model_performance_dir ):
        os.makedirs( model_performance_dir )

    for i in range(10000):
        estimator, eval_dict, current_step = train_2d_cnn(datadir, model_name)
        np.save( os.path.join( model_performance_dir, "{}_{}.npy".format( current_step, model_name ) ), eval_dict )
