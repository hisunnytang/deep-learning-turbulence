#!/home/kwoksun2/anaconda3/bin/python
import numpy
import matplotlib.pyplot as plt
from generate_3dIC import calculate_field_from_spectrum
from tqdm import tqdm

Nx = 512
Ny = 512
Nz = 512


#beta_range = numpy.linspace
beta_range = numpy.linspace( 0.1, 3, 200)[110:]
for beta in tqdm(beta_range):
	Dens = calculate_field_from_spectrum( beta, Nx, Ny, Nz )
	Dens = Dens.real
	Dens-= Dens.min()
	Dens /= Dens.sum()

	numpy.save( 'dens_data/dens_{0:0.2f}.npy'.format(beta), Dens )

