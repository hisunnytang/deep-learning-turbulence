#!/home/kwoksun2/anaconda3/bin/python

import numpy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import os

def convert_to_PPV(Dens,  V , Nx, Ny, Nv, vmin = -5.0, vmax = 5.0 ):
    ppv = np.zeros((Nx,Ny,Nv))
    bins = np.linspace( vmin, vmax ,  Nv + 1)

    for i in range(Nx):
        for j in range(Ny):
            count, _ = np.histogram(V[i,j,:], bins = bins, weights = Dens[i,j,:], density = False )
            ppv[i,j,:] = count.astype('float32')
    return ppv


vdata_path = './vdata'
ddata_path  = './dens_data'

dlist = numpy.linspace(0.1,3,200)
vlist = numpy.linspace(1.1,4,200)

def ppv_for_fixed_d(d):  
    print('parent process:', os.getppid())
    print('process id :', os.getpid(), 'with d = ', d)
    for v in vlist:
        outfilename = "ppvdata/ppv_d={0:0.4f}_v={1:0.4f}.npy".format( d, v )
        if os.path.isfile(outfilename): 
            print(outfilename, " exisit")
        else:
            print("generating ", outfilename)
            dfilename = ddata_path + '/dens_{0:0.2f}.npy'.format(d)
            vfilename = vdata_path  + '/{0:0.2f}_3dv.npy'.format(v)
            vdata = numpy.load(vfilename)
            ddata = numpy.load(dfilename)
            Nx, Ny, _ = vdata.shape
            NVCHANNELS = 128
        
            ppv = convert_to_PPV( ddata, vdata, Nx, Ny, NVCHANNELS, vmin = -5.0, vmax = 5.0 )
            numpy.save( outfilename  , ppv)
            del ppv
    print('parent process:', os.getppid())
    print('process id :', os.getpid(), 'done with d = ', d)
 
#for d in tqdm(dlist):
#    ppv_for_fixed_d(d)



pool = Pool()                      # create a multiprocessing Pool
pool.map( ppv_for_fixed_d, dlist ) # process output with pool
