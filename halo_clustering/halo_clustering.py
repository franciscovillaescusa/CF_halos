import numpy as np
import sys,os
import readfof
import MAS_library as MASL
import Pk_library as PKL

################################## INPUT ######################################
# location of the halo catalogues
root = '/mnt/ceph/users/fvillaescusa/Neutrino_simulations/Sims_Dec16_2/'

# parameters to select halos
BoxSize = 1000.0 #Mpc/h
Mmin    = 5e13   #Msun/h

# cosmology
model   = '0.0eV'
snapnum = 3 #3(z=0) 2(0.5) 1(z=1) 2(z=2)

# realization
realization = 1

# parameters for the Pk
grid = 256
###############################################################################

# find the name of the folder containing the halo catalogue
snapdir = '%s/%s/%d'%(root,model,realization)

# determine the redshift of the catalogue
z_dict = {3:0.0, 2:0.5, 1:1.0, 0:2.0}
redshift = z_dict[snapnum]

# read the halo catalogue
FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
                          swap=False, SFR=False, read_IDs=False)
										
# get the properties of the halos
pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h   
mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h           
vel_h = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s
Npart = FoF.GroupLen                #Number of CDM particles in the halo

# consider only the halos with M>Mmin
indexes = np.where(mass>Mmin)[0]
pos_h = pos_h[indexes]
mass  = mass[indexes]

# define the grid containing the halo density field
delta_h = np.zeros((grid,grid,grid), dtype=np.float32)

# assign halo positions to the grid
MASL.MA(pos_h, delta_h, BoxSize, MAS='CIC')
delta_h /= np.mean(delta_h, dtype=np.float64);  delta_h -= 1.0

# compute Pk
Pk = PKL.Pk(delta_h, BoxSize, axis=0, MAS='CIC', threads=1)

# get k and Pk
k  = Pk.k3D
Pk = Pk.Pk[:,0]
