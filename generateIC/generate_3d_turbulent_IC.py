import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import numpy
from os import listdir
import h5py
import numpy as np
from tqdm import tqdm
from generate_3dIC import *

beta_range = numpy.linspace(1.1 , 4.0, 200)[144:]

Nv = 128
NX = 512
NY = 512
NZ = 512

for i in tqdm(beta_range):
    print("getting PPV for beta =  {:0.2f}".format(i))
    V_x = calculate_3d_velocity( i, NX, NY, NZ)
    V_x1 = V_x.real - V_x.real.mean()
    vmin = numpy.min(V_x1)
    vmax = numpy.max(V_x1)
    ppv = convert_to_PPV(V_x1 , NX,NY,Nv, vmin = vmin, vmax = vmax)
    numpy.save("ppvdata/{:0.2f}_ppv.npy".format(i), ppv ) 
    numpy.save("vdata/{:0.02f}_3dv.npy".format(i), V_x1 ) 
    
fig, axes = plt.subplots(2, 2 )
fig.set_figheight(10)
fig.set_figwidth(10)
filenames = ["1.10_ppv.npy", "1.66_ppv.npy", "2.33_ppv.npy","3.15_ppv.npy" ]
axis_index = [(0,0),(0,1),(1,0),(1,1)]

for idx, filename in zip(axis_index, filenames):
    ppv = numpy.load('ppvdata/{}'.format(filename))
    x = numpy.random.randint(0,512)
    y = numpy.random.randint(0,512-128)
    im = axes[idx].imshow( ppv[x, y:y+128,: ], origin='lower', vmin=0, vmax=30 )
    beta = filename.strip('_ppv.npy')
    axes[idx].annotate(r'$\beta$ = '+beta, xy=(10,10) ,color='w'  )

    
fig.subplots_adjust(wspace=0.1, hspace=0.1)
cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig("sample_pvslice.png")


fig, axes = plt.subplots(2, 2 )
fig.set_figheight(10)
fig.set_figwidth(10)
filenames = ["1.10_ppv.npy", "1.66_ppv.npy", "2.33_ppv.npy","3.15_ppv.npy" ]
axis_index = [(0,0),(0,1),(1,0),(1,1)]

for idx, filename in zip(axis_index, filenames):
    ppv = numpy.load('ppvdata/{}'.format(filename))
    x = numpy.random.randint(0,512)
    y = numpy.random.randint(0,512-128)
    beta = filename.strip('_ppv.npy')
    im = axes[idx].plot( ppv[x, y,: ], label=r'$\beta$ = '+beta )
    axes[idx].legend()

fig.savefig("sample_pvslice.png")

