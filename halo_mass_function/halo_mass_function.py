import numpy as np
import sys,os
import readfof

################################## INPUT ######################################
# location of the halo catalogues
root = '/mnt/ceph/users/fvillaescusa/Neutrino_simulations/Sims_Dec16_2/'

# parameters for the halo mass function
BoxSize = 1000.0 #Mpc/h
Mmin    = 2e13   #Msun/h
Mmax    = 1e15   #Msun/h
bins    = 15

# cosmology
model   = '0.0eV'
snapnum = 3 #3(z=0) 2(0.5) 1(z=1) 2(z=2)

# realization
realization = 1
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

# get the bins for the halo mass function
bins_HMF = np.logspace(np.log10(Mmin), np.log10(Mmax), bins+1)

# get the size of the halo mass function intervals
dM = bins_HMF[1:] - bins_HMF[:-1]

# compute the halo mass function
HMF = np.histogram(mass, bins=bins_HMF)[0]/(dM*BoxSize**3)
