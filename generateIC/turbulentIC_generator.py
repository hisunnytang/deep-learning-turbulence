#!/home/kwoksun2/anaconda3/bin/python

import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import numpy
from os import listdir
import h5py
import numpy as np
from tqdm import tqdm
from generate_3dIC import *
import os
from multiprocessing import Pool
import gc
import multiprocessing
import itertools

beta_range  = numpy.linspace(1.1 , 4.0, 100)
gamma_range = numpy.linspace(1.1 , 4.0, 100)
NX = 512
NY = 512
NZ = 512


def generate_vfield(beta):
    filename = 'vdata/beta={:0.02f}.h5'.format(beta)
    if os.path.isfile(filename):
       print(filename, " exist")
    else:
       print("creating velocity for beta =  {:0.2f}".format(beta)," with ", multiprocessing.current_process() )
       vx = generate_random_field_from_spectrum( beta, NX, NY, NZ)
       vx /= vx.flatten().std()
       print('finished creating data ', multiprocessing.current_process() )
       f = h5py.File( filename , "w" )
       f.create_dataset('velocity',data=vx )
       f.close()
       print(filename, ' done')
    return None


def generate_dfield(gamma):
   filename = 'ddata/gamma={:0.02f}.h5'.format(gamma)
   if os.path.isfile(filename):
       print(filename, " exist" )
   else:
       print("creating density for beta =  {:0.2f}".format(gamma))
       rho = generate_random_field_from_spectrum( gamma, NX, NY, NZ )
       rho = rho - rho.min()
       rho/= rho.sum() / ( NX*NY*NZ)

       f = h5py.File( filename, "w" )
       f.create_dataset('density',data=rho )
       f.close()
       print(filename, ' done')
   return None

def generate_logdfield(gamma):
   # more realistic lognormal distribution
   # that follows the spectral index
   # with 0.1 to 10 density contrast
   filename = 'logddata/gamma={:0.02f}.h5'.format(gamma)
   if os.path.isfile(filename):
       print(filename, " exist" )
   else:
       print("creating density for beta =  {:0.2f}".format(gamma))
       rho = generate_random_field_from_spectrum( gamma, NX, NY, NZ )
       #rho /= rho.flatten().max()
       #rho *= 2.355
       #rho += 1.0

       rho /= rho.flatten().std()
       # this corresponds to the width of the log pdf
       # from Federrath et al. 2010 A&A 512, A81
       # sigma_s = 1.3 - 3
       # s = ln ( rho / rho_mean  )
       rho *= 2.0

       rho_lognorm = numpy.exp(rho)
       rho = rho_lognorm / rho_lognorm.mean()

       f = h5py.File( filename, "w" )
       f.create_dataset('density',data=rho )
       f.close()
       print(filename, ' done')
   return None

def convert_to_PPV(Dens,  V , Nx, Ny, Nv, vmin = -5.0, vmax = 5.0 ):
    ppv = np.zeros((Nx,Ny,Nv))
    bins = np.linspace( vmin, vmax ,  Nv + 1)

    for i in range(Nx):
        for j in range(Ny):
            count, _ = np.histogram(V[i,j,:], bins = bins, weights = Dens[i,j,:], density = False )
            ppv[i,j,:] = count.astype('float32')
    return ppv

def ppv_for_gamma_beta(gamma_beta ):
    gamma = gamma_beta[0]
    beta  = gamma_beta[1]
    outfilename = "log_ppvdata/ppv_ln_d={0:0.4f}_v={1:0.4f}.h5".format( gamma, beta )
    if os.path.isfile(outfilename):
        print(outfilename, " exisit")
    else:
        print("generating ", outfilename, " from ", os.getpid() )
        dfilename = 'logddata/gamma={0:0.2f}.h5'.format(gamma)
        vfilename = 'vdata/beta={0:0.2f}.h5'.format(beta)

        fd = h5py.File(dfilename,'r')
        ddata = fd['density'].value
        fd.close()

        fv = h5py.File(vfilename,'r')
        vdata = fv['velocity'].value
        fv.close()

        Nx, Ny, _ = vdata.shape
        NVCHANNELS = 64

        # sigma in the velocity field is 1
        # we include 4 sigma data points into our histogram
        # 2.355 / 8 * 64 ~ 20
        # the FWHM is resolved by ~20 points, if there are 64 channels

        ppv = convert_to_PPV( ddata, vdata, Nx, Ny, NVCHANNELS, vmin = -4.0, vmax = 4.0 )
        fppv = h5py.File(outfilename, 'w')
        fppv.create_dataset( 'ppv',   data=ppv   )
        fppv.create_dataset( 'gamma', data=gamma )
        fppv.create_dataset( 'beta',  data=beta  )
        fppv.close()
    print('process id :', os.getpid(), 'done with d = ', outfilename )

if __name__ == '__main__':
    pool = Pool(12, maxtasksperchild=1)
    pool.map( generate_logdfield,  gamma_range )
    pool.map( generate_vfield,  beta_range  )
#    pool.map( generate_dfield,  gamma_range )


    gamma_beta = [ gamma_range, beta_range ]
    gamma_beta_combo = list(itertools.product(*gamma_beta))
    np.random.shuffle( gamma_beta_combo )
    pool.map( ppv_for_gamma_beta, gamma_beta_combo  )


