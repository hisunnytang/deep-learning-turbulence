import numpy as np
cimport numpy as np



def generate_random_field_from_spectrum( np.float64_t beta, int Nx, int Ny, int Nz ):
    
    
    cdef np.ndarray[np.float64_t, ndim = 1] kx, ky, kz
    cdef np.ndarray[np.float64_t, ndim = 3] V_x, phase3d, gauss_amp, kx3d,ky3d,kz3d, Rk
    cdef np.ndarray[np.complex128_t, ndim = 3] Amp_K
    
    kx = np.fft.fftfreq( Nx, d = 1.0/Nx ) 
    ky = np.fft.fftfreq( Ny, d = 1.0/Ny ) 
    kz = np.fft.fftfreq( Nz, d = 1.0/Nz ) 
    
    # just to prevent the 1 / 0 error
    # no physical meaning
    kx[0] = 1.0e-10
    ky[0] = 1.0e-10
    kz[0] = 1.0e-10

    Amp_K = np.zeros((Nx,Ny,Nz),dtype=np.complex_)
    phase3d   = np.random.random((Nx,Ny,Nz)) * 3.14159 * 2.0
    # include gaussian fluctiation
    gauss_amp = np.random.normal(size=(Nx,Ny,Nz))     
 
    kx3d,ky3d,kz3d = np.meshgrid(kx,ky,kz)
    
    Rk = np.sqrt( kx3d*kx3d + ky3d*ky3d + kz3d*kz3d )
    Amp_K = Rk**( (- beta - 2.0)/2.0 ) * np.exp(phase3d*1j) * gauss_amp
    
    # this corresponds to the kmin
    cdef double kmin = 1.0
    Amp_K[Rk < kmin] = 0.0
    
    
    # normalize the Amp_K by the average dispersion
    Amp_K *= Nx*Ny*Nz
    

    # inverse fft will center the velocity field 
    V_x = np.fft.ifftn(Amp_K).real
    V_x/= 2.0 * np.pi
    
    return V_x

def calculate_power_spectrum(V_x, int  Rkbins = 128):
    
    Pk3d = np.abs(np.fft.fftn(V_x))**2.0
    
    Nx, Ny, Nz = Pk3d.shape
    
    kx = np.fft.fftfreq( Nx, d = 1.0/Nx ) 
    ky = np.fft.fftfreq( Ny, d = 1.0/Ny ) 
    kz = np.fft.fftfreq( Nz, d = 1.0/Nz )     
    kx3d,ky3d,kz3d = np.meshgrid(kx,ky,kz)
    
    Rk = np.sqrt( kx3d*kx3d + ky3d*ky3d + kz3d*kz3d )
    
    bins   = np.linspace( np.min(Rk), np.max(Rk), Rkbins )
    dbins  = bins[1] - bins[0]
    
    Pk_kR = []
    for i in range(Rkbins):
        kbin = bins[i]

        iloc = np.logical_and( (Rk > kbin) , (Rk < kbin+dbins) )
        Pk_kR.append( Pk3d[iloc].sum() )
    
    
    return bins, Pk_kR


def convert_to_PPV( V , Nx, Ny, Nv, vmin = -5.0, vmax = 5.0 ):
    ppv = np.zeros((Nx,Ny,Nv))
    bins = np.linspace( vmin, vmax ,  Nv + 1)

    for i in range(Nx):
        for j in range(Ny):
            count, _ = np.histogram(V[i,j,:], bins = bins)
            ppv[i,j,:] = count.astype('float32')
    return ppv
