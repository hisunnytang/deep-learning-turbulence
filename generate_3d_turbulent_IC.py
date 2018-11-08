import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import numpy
from os import listdir
import h5py
import numpy as np
cimport numpy as np

def calculate_3d_velocity( np.float64_t beta, int Nx, int Ny, int Nz  ):
    
    
    cdef np.ndarray[np.float64_t, ndim = 1] kx, ky, kz
    cdef np.ndarray[np.float64_t, ndim = 3] Amp_K, phase3d, kx3d,ky3d,kz3d, Rk
    cdef np.ndarray[np.complex128_t, ndim = 3] V_x
    
    kx = np.fft.fftfreq( Nx, d = 1.0/Nx ) 
    ky = np.fft.fftfreq( Ny, d = 1.0/Ny ) 
    kz = np.fft.fftfreq( Nz, d = 1.0/Nz ) 
    
    # just to prevent the 1 / 0 error
    # no physical meaning
    kx[0] = kx[1]*0.001
    ky[0] = ky[1]*0.001
    kz[0] = kz[1]*0.001

    Amp_K = np.zeros((Nx,Ny,Nz))
    phase3d = np.random.random((Nx,Ny,Nz)) * 3.14159 * 2.0
    
    kx3d,ky3d,kz3d = np.meshgrid(kx,ky,kz)
    
    Rk = np.sqrt( kx3d*kx3d + ky3d*ky3d + kz3d*kz3d )
    Amp_K = Rk**( (- beta - 2.0)/2.0 ) * np.cos(phase3d)
    
    # this corresponds to the kmin
    cdef double kmin = 3.0
    Amp_K[Rk < kmin] = 0.0
    
    # normalize the Amp_K by the average dispersion
    Vdisp = np.ones( (Nx,Ny,Nz) )*np.sqrt( kmin**(-beta + 1) / (beta - 1.0) )
    Amp_K = Amp_K / Vdisp
    
    V_x = np.fft.fftn(Amp_K)
    
    return V_x

def caculate_power_spectrum(V_x):
    
    Pk3d = numpy.abs(numpy.fft.fftn(V_x))**2.0
    
    Nx, Ny, Nz = Pk3d.shape
    
    kx = np.fft.fftfreq( Nx, d = 1.0/Nx ) 
    ky = np.fft.fftfreq( Ny, d = 1.0/Ny ) 
    kz = np.fft.fftfreq( Nz, d = 1.0/Nz )     
    kx3d,ky3d,kz3d = np.meshgrid(kx,ky,kz)
    
    Rk = np.sqrt( kx3d*kx3d + ky3d*ky3d + kz3d*kz3d )
    
    Rkbins = 128
    bins   = np.linspace( np.min(Rk), np.max(Rk), Rkbins )
    dbins  = bins[1] - bins[0]
    
    Pk_kR = []
    for i in range(Rkbins):
        kbin = bins[i]

        iloc = numpy.logical_and( (Rk > kbin) , (Rk < kbin+dbins) )
        Pk_kR.append( Pk3d[iloc].sum() )
    
    
    return bins, Pk_kR


def convert_to_PPV( V , Nx, Ny, Nv, vmin = -5.0, vmax = 5.0 ):
    ppv = numpy.zeros((Nx,Ny,Nv))
    bins = numpy.linspace( vmin, vmax ,  Nv + 1)

    for i in range(Nx):
        for j in range(Ny):
            count, _ = numpy.histogram(V[i,j,:], bins = bins)
            ppv[i,j,:] = count.astype('float32')
    return ppv



beta_range = numpy.linspace(1.1 , 4.0, 200)

Nv = 128
NX = 512
NY = 512
NZ = 512

for i in beta_range:
    print("getting PPV for beta =  {:0.2f}".format(i))
    V_x = calculate_3d_velocity( i, NX, NY, NZ)
    V_x1 = V_x.real - V_x.real.mean()
    vmin = numpy.min(V_x1)
    vmax = numpy.max(V_x1)
    ppv = convert_to_PPV(V_x1 , NX,NY,Nv, vmin = vmin, vmax = vmax)
    
    hf = h5py.File("ppvdata/{:0.2f}_ppv.npy".format(i) , "w")
    hf.create_dataset( 'ppv', data = ppv )
    hf.close()
    
    
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

