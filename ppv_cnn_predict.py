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
from ppv_cnn_input import get_slices_from_ppv
from ppv_cnn_train import model_name
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)


beta_range  = numpy.linspace(1.1 , 4.0, 100)
gamma_range = numpy.linspace(0.1 , 3.0, 100)  

def validate_on_ppv_cube( filename, N = 10000 ):


    gpu_options = tf.GPUOptions( allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig(session_config = sess_config)  
    beta_estimator = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir=model_name, config=run_config)


    X_valid, Y_valid = get_slices_from_ppv(filename, N = N)
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

if __name__ == "__main__":
     NBETA  = 5
     NGAMMA = 5

     f, axarr = plt.subplots( NBETA,NGAMMA, figsize=(10,10), gridspec_kw = {'wspace':0, 'hspace':0})
     for i, beta in enumerate(beta_range[:: 100//NBETA  ]):
         for j, gamma in enumerate(gamma_range[::100//NGAMMA ]):
             fn = "generateIC/ppvdata/ppv_d={0:0.4f}_v={1:0.4f}.h5".format( gamma, beta )
             _, predictions, current_step = validate_on_ppv_cube(fn, N = 1000)
             ax = axarr[i,j].hist( predictions['beta_pred'],color='C0', bins=beta_range)
             ax = axarr[i,j].hist( predictions['gamma_pred'],color='C1' , bins = gamma_range)
             ax = axarr[i,j].axvline( predictions['beta'][0],ls='--', color='C0',label=r'true $\beta$' )
             ax = axarr[i,j].axvline( predictions['gamma'][0],ls='--', color='C1',label=r'true $\gamma$' )
     plt.tight_layout()
     plt.legend()
     f.savefig("{}_performance_on_ppv.png".format(current_step) )

    
