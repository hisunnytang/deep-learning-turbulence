#!/home/kwoksun2/anaconda3/bin/python

import numpy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import os
import h5py

def convert_to_PPV(Dens,  V , Nx, Ny, Nv, vmin = -5.0, vmax = 5.0 ):
    ppv = np.zeros((Nx,Ny,Nv))
    bins = np.linspace( vmin, vmax ,  Nv + 1)

    for i in range(Nx):
        for j in range(Ny):
            count, _ = np.histogram(V[i,j,:], bins = bins, weights = Dens[i,j,:], density = False )
            ppv[i,j,:] = count.astype('float32')
    return ppv


vdata_path = './vdata'
ddata_path  = './ddata'

dlist = numpy.linspace(0.1,3.0,100)
vlist = numpy.linspace(1.1,4.0,100)

def ppv_for_fixed_d(d):  
    print('parent process:', os.getppid())
    print('process id :', os.getpid(), 'with d = ', d)
    for v in vlist:
        outfilename = "ppvdata/ppv_d={0:0.4f}_v={1:0.4f}.h5".format( d, v )
        if os.path.isfile(outfilename): 
            print(outfilename, " exisit")
        else:
            print("generating ", outfilename)
            dfilename = ddata_path + '/dens_{0:0.2f}.npy'.format(d)
            vfilename = vdata_path  + '/{0:0.2f}_3dv.npy'.format(v)
            
            fd = h5py.File(dfilename,'r')
            ddata = fd['density'].values
            fd.close()

            fv = h5py.File(vfilename,'r')
            vdata = fv['velocity'].values
            fv.close()

            Nx, Ny, _ = vdata.shape
            NVCHANNELS = 64
        
            ppv = convert_to_PPV( ddata, vdata, Nx, Ny, NVCHANNELS, vmin = -5.0, vmax = 5.0 )
            fppv = h5py.File(outfilename, 'w')
            fppv.create_dataset( 'ppv', data=ppv )
            fppv.close()
    print('parent process:', os.getppid())
    print('process id :', os.getpid(), 'done with d = ', d)
 
#for d in tqdm(dlist):
#    ppv_for_fixed_d(d)



pool = Pool()                      # create a multiprocessing Pool
pool.map( ppv_for_fixed_d, dlist ) # process output with pool
