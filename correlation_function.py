from mpi4py import MPI
import numpy as np
import readfof
import correlation_function_library as CFL
import redshift_space_library as RSL
import random, sys, os

# MPI definitions
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

###################################### INPUT ###################################
root     = '/mnt/ceph/users/fvillaescusa/Neutrino_simulations/Sims_Dec16_2/'
snapnum  = 4
redshift = 0
Mmin     = 1e14 #Msun/h
Rmin     = 1.0  #Mpc/h
Rmax     = 50.0 #Mpc/h

#### RANDOM CATALOG ####
random_points = 1000000  #number of points in the random catalogue

#### PARTIAL RESULTS NAMES ####
DD_name = 'DD.dat';   DR_name = 'DR.dat'

#### CF PARAMETERS ####
BoxSize  = 1000.0  #Mpc/h
bins     = 30    #number of bins in the CF
Rmin     = 1.0   #Mpc/h
Rmax     = 50.0  #Mpc/h

#### MODEL ####
model = '0.0eV'
################################################################################

# do a loop over all realizations
for i in xrange(1,101):

    # do a loop over the different models
    for model in ['0.0eV', '0.10eV_degenerate', '0.15eV',
                  '0.0eV_0.798', '0.0eV_0.807', '0.0eV_0.818', '0.0eV_0.822']:

        folder_out = '%s'%model
        if not(os.path.exists(folder_out)):  os.system('mkdir %s'%folder_out)

        # do a loop over the different halo masses
        for Mmin in [5e13, 7e13, 1e14, 3e14]:

            # get the name of the output file and the snapdir
            snapdir = '%s/%s/%d'%(root,model,i)
            f_out   = '%s/CF_%s_%.2e_%d.txt'%(folder_out,model,Mmin,i)
            if os.path.exists(f_out):  continue

            # get the positions of the random particles
            pos_r, RR_name = CFL.create_random_catalogue(random_points, Rmin, 
                                                         Rmax, bins, BoxSize)
            
            # we set here the actions
            DD_action = 'compute'
            RR_action = 'read'  #if needed, the RR pairs are computed above
            DR_action = 'compute'
            
            # Only the master will read the positions of the galaxies
            pos_h = None   

            #### MASTER ####
            if myrank==0:

                #read FoF-halos/subfind-halos/subhalos information    
                print '\nReading galaxy catalogue'

                # read the halo catalogue
                FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
                                          swap=False, SFR=False, read_IDs=False)
										
                # get the properties of the halos
                pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h
                mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h
                vel_h = FoF.GroupVel*(1.0+redshift) #Halo velocities in km/s

                # only consider halos above Mmin
                indexes = np.where(mass>Mmin)[0]
                pos_h   = pos_h[indexes]


            # compute the 2pt correlation function
            if myrank==0:  print '\nComputing the 2pt-correlation function...'

            r, xi_r, error_xi = CFL.TPCF(pos_h,     pos_r,     BoxSize,
                                         DD_action, RR_action, DR_action,
                                         DD_name,   RR_name,   DR_name,
                                         bins,      Rmin,      Rmax)
                            
            # save results to file
            if myrank==0:  np.savetxt(f_out, np.transpose([r, xi_r, error_xi]))
            

