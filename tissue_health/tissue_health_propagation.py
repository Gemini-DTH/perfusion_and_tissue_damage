"""
Estimate the infarct volume using a tissue health model with vulnerability propagation

Yidan Xue - 2022/02

"""

from dolfin import *
import scipy.interpolate
import argparse
import time
import yaml
import numpy as np
import os
import copy
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import multiprocessing

# added module
import IO_fcts
import finite_element_fcts as fe_mod

start0 = time.time()

# %% READ INPUT
if rank == 0: 
    print('Step 1: Reading the input')

parser = argparse.ArgumentParser(description="infarct computation based on perfusion results")
parser.add_argument("--config_file", help="path to configuration file",
                    type=str, default='./config_propagation.yaml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                type=str, default=None)
config_file = parser.parse_args().config_file

configs = IO_fcts.basic_flow_config_reader_yml(config_file,parser)

healthyfile = configs['input']['healthyfile']
strokefile = configs['input']['strokefile']
treatmentfile = configs['input']['treatmentfile']

arrival_time = configs['input']['arrival_time']
recovery_time = configs['input']['recovery_time']

res_fldr = configs['output']['res_fldr']

kf,kt,kc,kd = configs['parameter']['kf'],configs['parameter']['kt'],configs['parameter']['kc'],configs['parameter']['kd']
Td,Tp = configs['parameter']['Td'],configs['parameter']['Tp']

# define the ODE for cell death, h[0] - dead, h[1] - toxic, h[2] - vulnerable index
def cell_death(x,t):
	D,T,V = x
	A = 1-D

	if T>Td:
		dDdt = kf*A*T
	else:
		dDdt = 0
	dTdt = kt*V*(1-T)-kc*(1-V)*A*T

	dxdt = [dDdt, dTdt, 0]
	return dxdt

# %% READ PERFUSION
if rank == 0: 
    print('Step 2: Reading perfusion files')

# read healthy perfusion image
img_h = nib.load(healthyfile)
header_h = img_h.header
affine_matrix = np.eye(4)
affine_matrix[0,:] = header_h.get("srow_x")
affine_matrix[1,:] = header_h.get("srow_y")
affine_matrix[2,:] = header_h.get("srow_z")
img_h = img_h.get_fdata()

# read occluded perfusion image
img_o = nib.load(strokefile)
img_o = img_o.get_fdata()

# read treatment perfusion image
img_t = nib.load(treatmentfile)
img_t = img_t.get_fdata()

# compute the relative perfusion
rel_perf = img_o/img_h
rel_perf[np.isnan(rel_perf)] = 0

# mask the brain region and initialise the model
brain_region = np.int32(img_h > 0)
nx,ny,nz = rel_perf.shape

# compute the relative perfusion after treatment
rel_perf_t = img_t/img_h
rel_perf_t[np.isnan(rel_perf_t)] = 0

# %% RUN CELL DEATH MODEL
if rank == 0:
    print('Step 3: Running the cell death model')

start1 = time.time()

# BEFORE TREATMENT

dt = 30 # min
t = np.linspace(0,dt*60,int(dt*12+1))
num_iter_arrival = int(arrival_time*60/dt)

infarct = np.zeros([nx,ny,nz])
toxin = np.zeros([nx,ny,nz])
vulnerable = np.zeros([nx,ny,nz])

# initialise vulnerable index
for i in range(nx):
	for j in range(ny):
		for k in range(nz):
			if brain_region[i,j,k]!=0:
				if rel_perf[i,j,k]<1:
					vulnerable[i,j,k] = 1-rel_perf[i,j,k]

for n in range(num_iter_arrival):

	# define an array to update vulnerable index
	vulnerable_new = np.zeros([nx,ny,nz])

	# run cell death model first
	for i in range(nx):
		for j in range(ny):
			for k in range(nz):
				if brain_region[i,j,k] != 0:
					# just run cell death model for the progress region
					# if vulnerable[i,j,k] > 0 and (infarct[i,j,k] <= 0.8 or toxin[i,j,k] <= Tp):
					# 	hi = [infarct[i,j,k],toxin[i,j,k],vulnerable[i,j,k]]
					# 	hs = odeint(cell_death, hi, t)
					# 	infarct[i,j,k] = hs[-1,0]
					# 	toxin[i,j,k] = hs[-1,1]
					# or run for every pixel
					hi = [infarct[i,j,k],toxin[i,j,k],vulnerable[i,j,k]]
					hs = odeint(cell_death, hi, t)
					infarct[i,j,k] = hs[-1,0]
					toxin[i,j,k] = hs[-1,1]

    # run the vulnerable index propagation
	for i in range(nx):
		for j in range(ny):
			for k in range(nz):
				if brain_region[i,j,k] != 0 and toxin[i,j,k]>Tp:
					vulnerable_new[i+1,j,k] = max(vulnerable[i,j,k]*kd,vulnerable[i+1,j,k],vulnerable_new[i+1,j,k])
					vulnerable_new[i-1,j,k] = max(vulnerable[i,j,k]*kd,vulnerable[i-1,j,k],vulnerable_new[i-1,j,k])	
					vulnerable_new[i,j+1,k] = max(vulnerable[i,j,k]*kd,vulnerable[i,j+1,k],vulnerable_new[i,j+1,k])
					vulnerable_new[i,j-1,k] = max(vulnerable[i,j,k]*kd,vulnerable[i,j-1,k],vulnerable_new[i,j-1,k])
					vulnerable_new[i,j,k+1] = max(vulnerable[i,j,k]*kd,vulnerable[i,j,k+1],vulnerable_new[i,j,k+1])
					vulnerable_new[i,j,k-1] = max(vulnerable[i,j,k]*kd,vulnerable[i,j,k-1],vulnerable_new[i,j,k-1])

	# update the vulnerable index
	for i in range(nx):
		for j in range(ny):
			for k in range(nz):
				if brain_region[i,j,k] != 0 and vulnerable_new[i,j,k] != 0:
					vulnerable[i,j,k] = vulnerable_new[i,j,k]	

# AFTER TREATMENT - assume successful treatment (vulnerability based on restored perfusion)

# update vulnerable index

vulnerable = np.zeros([nx,ny,nz])

for i in range(nx):
	for j in range(ny):
		for k in range(nz):
			if brain_region[i,j,k]!=0:
				if rel_perf_t[i,j,k]<1:
					vulnerable[i,j,k] = 1-rel_perf_t[i,j,k]

# run the cell death model again

t = np.linspace(0,recovery_time*3600,int(recovery_time*60+1))

for i in range(nx):
	for j in range(ny):
		for k in range(nz):
			if brain_region[i,j,k] != 0:
				hi = [infarct[i,j,k],toxin[i,j,k],vulnerable[i,j,k]]
				hs = odeint(cell_death, hi, t)
				infarct[i,j,k] = hs[-1,0]
				toxin[i,j,k] = hs[-1,1]

# calculate the core volume
core = np.sum(infarct>0.8)*pow(affine_matrix[0,0],3)/1000

end1 = time.time()

if len(sys.argv) >= 2:
    # The second argument indicates the path where to write a summary of
    # outcome parameters too, this now considers only the infarct core volume.
    with open(sys.argv[2], 'w') as outfile:
        yaml.safe_dump(
            {'core-volume': core},
            outfile
        )

end0 = time.time()

if rank == 0:
    print('The core volume is '+str(core)+' mL')
    print('Infarct computation time [s]; \t\t\t', end1 - start1)
    print('Simulation finished - Total execution time [s]; \t\t\t', end0 - start0)
