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
tf.logging.set_verbosity(tf.logging.INFO)


# In[11]:


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # input layer should be of shape [:, NXCHANNELS, NVCHANNELS, 1]
    # NVCHANNELS: number of velocity bins
    
    NVCHANNELS=64
    NXCHANNELS=64
    
    input_layer = tf.reshape(features["x"], [-1,NXCHANNELS, NVCHANNELS, 1])
    
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
    

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 4x4x32]
    # Output Tensor Shape: [batch_size, 32]
    dense = tf.layers.dense(inputs=pool3_flat, units=32, activation=tf.nn.relu)
    
    # Add dropout operation; 0.7 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
    
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 1]
    
###beta_pred, gamma_pred = tf.layers.dense(inputs=dropout, units=2)
    
    logits = tf.layers.dense(inputs=dropout, units=2)
    print(logits, logits.shape)
    beta_pred =  logits[:,0]
    gamma_pred = logits[:,1]

    
    # Normalize the final layer to the range of beta
    # A simple Ax + b layer
    predictions = { "beta_pred": beta_pred, "gamma_pred": gamma_pred }
   
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)   
    # Simple Mean Squared 
    loss_gamma = tf.losses.mean_squared_error( labels['gamma'] , predictions['gamma_pred'] )
    loss_beta  = tf.losses.mean_squared_error( labels['beta'] , predictions['beta_pred']  )
    loss       = loss_beta + loss_gamma
 
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    gamma_accuracy = tf.metrics.accuracy( labels['gamma'], predictions['gamma_pred'] )
    beta_accuracy  = tf.metrics.accuracy( labels['beta'],  predictions['beta_pred'] )

    eval_metric_ops = { "beta_accuracy": beta_accuracy, "gamma_accuracy": gamma_accuracy }
    return tf.estimator.EstimatorSpec( mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def random_slice_index():
    """Create slicing index for the PPV cube
    Returns:
      retain_x:    Boolean. Decide if x/y axis is kept
      left/ right: Slicing index to match desire output size (NSPATIAL)
      sliceidx:    Index of the layer to be taken
    """
    left  = np.random.randint( 0, 512 - 64 )
    right = left + 64
    sliceidx = np.random.randint( 0, 512 )
    retain_x = bool( np.random.randint( 0, 2  ) )
    return retain_x , left, right, sliceidx


def read_slices_hdf5(filename):
    """Read PV slice from hdf5 file
    Args: 
      filename: location of the hdf5 file
    Return:
      image: postion-velocity slice: a 2D array with size [NSPATIAL, NV]
      label:
    """
    with h5py.File( filename, 'r' ) as hf:
        retain_x, left, right, sliceidx = random_slice_index()
        im = hf["ppv"]
        gamma = hf['gamma'].value
        beta  = hf['beta'].value
        retain_x, left, right, sliceidx = random_slice_index() 
        if retain_x:
            image = im[left:right, sliceidx,:]
        else:
            image =  im[sliceidx, left:right,:]
    return image, gamma, beta

def get_batch( filenames, N = 128, dims = (64, 64)):
    """
    Create the PV slice from the PPV cube
    """
    
    numpy.random.shuffle(filenames)   
   
    X = []
    Yd = []
    Yv = []
    count = 0
    for idx in range(N):
        filename = filenames[idx]
        print(filename, count) 

        pvslice, gamma, beta = read_slices_hdf5(filename)
       
        X.append(pvslice)
        Yd.append(gamma)
        Yv.append(beta)        

        count += 1
        #del ppvdata
        #del pvslice
        #gc.collect()
    X = np.array(X)
    Yd = np.array(Yd)
    Yv = np.array(Yv)
    Y = {}
    
    Y['beta']  = Yv
    Y['gamma'] = Yd

    return X,Y


def train_2d_cnn():
    # handcrafted training sample

    filenames = glob.glob("/home/kwoksun2/deep-learning-turbulence/generateIC/ppvdata/*h5")
    filenames.sort(key=os.path.getmtime) 
    # avoid reading data that havent finished generating
    filenames = filenames[:-6] 

    X, Y = get_batch(filenames, N = 256)
    
    # Create the Estimator
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    gpu_options = tf.GPUOptions( allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig(session_config = sess_config)  
    beta_estimator = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir="./turbulence_2dconvnet_model", config=run_config)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    #     tensors_to_log = {"beta": "softmax_tensor"}
    #     logging_hook = tf.train.LoggingTensorHook(
    #       tensors=tensors_to_log, every_n_iter=50)



    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X},
        y=Y,
        batch_size=16,
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
          model_fn=cnn_model_fn, model_dir="./turbulence_2dconvnet_model", config=run_config)
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


# In[ ]:


for i in range(5000):
    _, eval_dict, current_step = train_2d_cnn() 
    np.save( "../{}_2d_convnet.npy".format(current_step), eval_dict )

# In[6]:


X,y = get_batch(N=10)

# Create the Estimator
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
run_config = tf.estimator.RunConfig(session_config = sess_config)  
beta_estimator = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./turbulence_2dconvnet_model", config=run_config)


beta_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X},
    y=y[:,None],
    num_epochs=1,
    shuffle=False)

beta_predict = beta_estimator.predict(input_fn=beta_input_fn)


# In[7]:


tf_predicted_data = []
for bbb in beta_predict:
    tf_predicted_data.append(bbb['beta_pred'])


# In[8]:


import numpy
numpy.array(tf_predicted_data).shape


# In[10]:


import matplotlib.pyplot as plt

plt.scatter( y,  tf_predicted_data )
plt.plot( range(1,6), range(1,6), color='r' )

beta_range = numpy.arange(1,6)
plt.fill_between( beta_range , 0.9*beta_range , 1.1*beta_range, alpha = 0.5  )
plt.xlim(1,4.5)
plt.ylim(1,4.5)
plt.axes().set_aspect('equal')



