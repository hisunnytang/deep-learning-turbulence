#!/usr/bin/python3
# coding: utf-8


"""Reading Slices from premade hdf5 files"""

import numpy
import numpy as np
from os import listdir
import gc
import os
import glob
import h5py
from scipy.ndimage import gaussian_filter

def random_slice_index( ncube = 512, nspatial = 64  ):
    """Create slicing index for the PPV cube
    Args:
      ncube:    physical dimension of the ppv cube
      nspatial: desired output pvslice dimension
    Returns:
      retain_x:    Boolean. Decide if x/y axis is kept
      left/ right: Slicing index to match desire output size (NSPATIAL)
      sliceidx:    Index of the layer to be taken
    """
    left  = np.random.randint( 0, ncube - nspatial )
    right = left + nspatial
    sliceidx = np.random.randint( 0, ncube )
    retain_x = bool( np.random.randint( 0, 2  ) )
    return retain_x , left, right, sliceidx

def add_gaussian_noise( image, SNR ):
    peak  = image.max()
    noise = np.random.normal( loc=0.0, scale = peak / SNR,  size = (image.shape) )
    return noise + image

    
   

def read_slices_hdf5(filename, add_noise = False, gaussian_smoothing = False):
    """Read PV slice from hdf5 file
    Args: 
      filename: location of the hdf5 file
    Return:
      image: postion-velocity slice: a 2D array with size [NSPATIAL, NV]
      label:
    """
    with h5py.File( filename, 'r' ) as hf:
        im = hf["ppv"]
        gamma = hf['gamma'].value
        beta  = hf['beta'].value
        if not gaussian_smoothing:
            retain_x, left, right, sliceidx = random_slice_index() 
            if retain_x:
                image = im[left:right, sliceidx,:]
            else:
                image =  im[sliceidx, left:right,:]
        else:
            image = np.zeros((64,64))
            # in pixel units
            # size = 3 => 3x3 filter
            size_of_filter = numpy.random.choice( [1,3,5,7] )
            # sigma is chosen such that FWHM = 2.33 sigma = size of filter
            sigma = size_of_filter / 2.355
            retain_x = bool( np.random.randint(0,2) )
            elements_needed = size_of_filter * 64
            # df: half the filter size
            df = int(np.floor(size_of_filter/2.0))

            slice_index1 = np.random.randint( 0, 512 - size_of_filter)
            slice_index2 = np.random.randint( 0, 512 - elements_needed) 
            print("filter size =", size_of_filter, "; slice_ind:", slice_index1, slice_index2)
            if retain_x:
                ppv_partial = im[ slice_index2 : slice_index2 + elements_needed , slice_index1: slice_index1+size_of_filter, :] 
                for i in range(image.shape[1]):
                    image[:,i] = gaussian_filter( ppv_partial[:,:,i], sigma = sigma )[ df::size_of_filter, df]
            else:
                ppv_partial = im[ slice_index1: slice_index1+size_of_filter, slice_index2: slice_index2+elements_needed, : ]
                for i in range(image.shape[1]):
                    image[:,i] = gaussian_filter( ppv_partial[:,:,i], sigma = sigma )[df, df::size_of_filter]
 
    if add_noise:
        SNR = 17.0 * np.random.random_sample() + 3.0
        print('with SNR = ', SNR)
        noisy_image = add_gaussian_noise( image, SNR ) 
        return  noisy_image, gamma, beta
    else:
        return image, gamma, beta

def get_batch( filenames, N = 128, dims = (64, 64)):
    """
    Create the PV slice from randomly selected PPV cubes
    """
    
    numpy.random.shuffle(filenames)   
   
    X = []
    Yd = []
    Yv = []
    count = 0
    for idx in range(N):
        filename = filenames[idx]
        print(filename, count) 

        pvslice, gamma, beta = read_slices_hdf5(filename, add_noise=True,gaussian_smoothing=True)
       
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

def get_slices_from_ppv( filename, N = 10000, dims = (64, 64), add_noise=False, gaussian_smoothing=False):
    """
    Create the PV slice from 1 PPV cube
    """
    
    X = []
    Yd = []
    Yv = []
    count = 0
    for idx in range(N):

        pvslice, gamma, beta = read_slices_hdf5(filename, add_noise=add_noise, 
                                                gaussian_smoothing=gaussian_smoothing)
        X.append(pvslice)
        Yd.append(gamma)
        Yv.append(beta)        

        count += 1
    X = np.array(X)
    Yd = np.array(Yd)
    Yv = np.array(Yv)
    Y = {}
    
    Y['beta']  = Yv
    Y['gamma'] = Yd

    return X,Y


