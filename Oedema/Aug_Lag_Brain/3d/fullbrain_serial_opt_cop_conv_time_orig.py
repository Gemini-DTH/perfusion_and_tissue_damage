#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from dolfin import *
import copy
import time
import meshio
import os 
import finite_element_fcts as fe_mod
import pandas as pd

os.environ['OMP_NUM_THREADS'] = '1'
t0=time.time()
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

#Parameters for Poroelastic Model
p_element_degree = 1#degree
u_element_degree = 1#degree
p_baseline_val = 1330#normal interstitial pressure
#p_blood_val = 6000#feeding blood pressure 
nu = 0.35#poisson ratio
G = 216.3*2*1.37 #*2*2*1.28 #shear modulus
viscosity_blood = 3.6E-3#blood viscosity pa.s
k_water = 3.6E-15#water conductivity
viscosity_water = 1E-3#water viscosity
u_vent = Constant((0.0, 0.0, 0.0)) 
mu = Constant(2*G)   	
lmbda = Constant(2*G/(1.0 - 2.0*nu))
Q_water = 3244
Q_blood = 628
T = 10000            # final time
num_steps = 30   # number of time steps
dt = T / num_steps # time step size
k = Constant(dt)
num_steps = 6
#Create Mesh, Labels and bbtrees
L_p = 3E-11*1 # !!!!!!!!!!11 #3e-10 # 
tao = 0.65 #.13  # changed
os_p = 2445 #+ 785
E_cap =  864.5 
R_cap = 5E-6 
n_b = 0.03 

k_blood =  3.6e-13 # ##ten times!!!!! 1E-14 #*1e-5   #1e-15 # #1e-12  delta 50   1e-13 200 (wahbi)   7大10小      
k_blood_a = 4.4424e-12 
k_blood_v = k_blood_a*2 
trac = 1.326e-6  #3e-6#
trcv = 4.641e-6  #trac*2#

ABP = 133.32*90

mesh = Mesh("mesh.xml")
md = MeshFunction("size_t", mesh, "mesh_func.xml") #size_t
mf = MeshFunction("size_t", mesh, "mesh_funcf.xml")
Vk = FunctionSpace(mesh, "Lagrange", 1)  
v_expr = Expression('1', degree=1)
v_exe = interpolate(v_expr, Vk)

K = fe_mod.tensor_generate(mesh,md, mf)

#Oedema
dx = Measure('dx', domain=mesh, subdomain_data=md)
ds = Measure('ds', domain=mesh, subdomain_data=mf)

#Define BCs and variational problem
Vp = FunctionSpace(mesh, 'CG', p_element_degree)
element_type = mesh.ufl_cell()
cgpw = FiniteElement('CG', element_type, 1)
cgpb = FiniteElement('CG', element_type, 1)
cgpa = FiniteElement('CG', element_type, 1)
cgpv = FiniteElement('CG', element_type, 1)
element = MixedElement([cgpw, cgpb, cgpa, cgpv]) 
V = FunctionSpace(mesh, element)
fcts = TrialFunction(V)
vpw, vpb, vpa, vpv = TestFunction(V)
pw, pb, pa, pv= split(fcts)

Vu = VectorFunctionSpace(mesh, 'CG', u_element_degree)

bcsp = []
for i in [1,2,3,4,5,6,8]:
   a = i #+ 21
   bc = DirichletBC(V.sub(0), Constant(1330), mf, a)
   bcsp.append(bc)
   bcsp.append(bc)
bc = DirichletBC(V.sub(0), Constant(1330), mf, 7)
bcsp.append(bc)

#bcspa = []
for i in [1,2,3,4,5,6,8]:
   a = i #+ 21
   bc = DirichletBC(V.sub(2), Constant(ABP), mf, a)
   bcsp.append(bc)


#bcspv = []
for i in [1,2,3,4,5,6,8]:
   a = i #+ 21
   bc = DirichletBC(V.sub(3), Constant(ABP-10000), mf, a)
   bcsp.append(bc)

bcs_fluid = bcsp
pw, pb, pa, pv= split(fcts)
LHSb = (k_blood/viscosity_blood)*dot(grad(pb), grad(vpb))*dx # 
RHSb =  trcv*(pb-pv)*vpb*dx -  trac*(pa-pb)*vpb*dx 
LHSa =  (k_blood_a/viscosity_blood)*dot(K*grad(pa), grad(vpa))*dx #
RHSa =  trac*(pa-pb)*vpa*dx 
LHSv =  (k_blood_v/viscosity_blood)*dot(K*grad(pv), grad(vpv))*dx # 
RHSv =  - trcv*(pb-pv)*vpv*dx 
LHSw = (k_water/viscosity_water)*dot(grad(pw), grad(vpw))*dx 
F = LHSb + RHSb + LHSw + RHSa + LHSa + LHSv + RHSv 
RHS = rhs(F) 
LHS = lhs(F) 
lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False
fcts = fe_mod.solve_lin_sys(V,LHS,RHS,bcsp,lin_solver,precond,rtol,mon_conv,init_sol)
(pw_n, pb_n, pa_n, pv_n) = fcts.split(True)

file = File('pan.pvd')
file << pa_n
file = File('pbn.pvd')
file << pb_n
file = File('pvn.pvd')
file << pv_n
file = File('pwn.pvd')
file << pw_n
vpw, vpb, vpa, vpv = TestFunction(V)

bcsp = []
for i in [1,2,3,4,5,6,8]:
   a = i #+ 21
   bc = DirichletBC(V.sub(0), Constant(1330), mf, a)
   bcsp.append(bc)
   bcsp.append(bc)
bc = DirichletBC(V.sub(0), Constant(1330), mf, 7)
bcsp.append(bc)

#bcspa = []
for i in [1,2,3,4,5,6,8]:
   a = i #+ 21
   bc = DirichletBC(V.sub(2), Constant(ABP), mf, a)
   bcsp.append(bc)
   bcsp.append(bc)

#bcspv = []
for i in [1,2,3,4,5,6,8]:
   a = i #+ 21
   bc = DirichletBC(V.sub(3), Constant(ABP-10000), mf, a)
   bcsp.append(bc)
   bcsp.append(bc)

bcs_fluid = bcsp

n = FacetNormal(mesh)
pressure_max = []
time_step = 0
print('initiation done')

num_steps = 8

while time_step < num_steps: 
   mls = []
   u_magnitude_max = 1.0
   distance_ave = 1
   step = 0
   contact_vertices = [1,1]
   while step<6:
      fcts = TrialFunction(V)
      pw, pb, pa, pv = split(fcts)
      LHSb = (pb-pb_n)*vpb/k/Q_blood*dx + (k_blood/viscosity_blood)*dot(grad(pb), grad(vpb))*dx # 
      RHSb =  trcv*(pb-pv)*vpb*dx -  trac*(pa-pb)*vpb*dx + 2*n_b*(L_p/R_cap)*(pb - pw - tao*os_p)*vpb*dx(50)

      LHSa =  (pa-pa_n)*vpa/k/Q_blood*dx + (k_blood_a/viscosity_blood)*dot(K*grad(pa), grad(vpa))*dx #
      RHSa =  trac*(pa-pb)*vpa*dx 

      LHSv =  (pv-pv_n)*vpv/k/Q_blood*dx + (k_blood_v/viscosity_blood)*dot(K*grad(pv), grad(vpv))*dx # 
      RHSv =  - trcv*(pb-pv)*vpv*dx 

      LHSw = (k_water/viscosity_water)*dot(grad(pw), grad(vpw))*dx + (pw-pw_n)/k/Q_water*vpw*dx
      RHSw = 2*n_b*(L_p/R_cap)* (pb - pw - tao*os_p)*vpw*dx(50) 

      F = LHSb + RHSb - RHSw + LHSw + RHSa + LHSa + LHSv + RHSv 
      RHS = rhs(F) 
      LHS = lhs(F) 
      lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False
      fcts = fe_mod.solve_lin_sys(V,LHS,RHS,bcsp,lin_solver,precond,rtol,mon_conv,init_sol)
      (pw, pb, pa, pv) = fcts.split(True)

      pa1 = project(pa, Vp, solver_type='bicgstab', preconditioner_type='amg')
      pb1 = project(pb, Vp, solver_type='bicgstab', preconditioner_type='amg')
      pw1 = project(pw, Vp, solver_type='bicgstab', preconditioner_type='amg')
      pv1 = project(pv, Vp, solver_type='bicgstab', preconditioner_type='amg')
      pmax_loc = pw1.vector().get_local().max()
      comm.barrier()
      pmax = MPI.max(comm, pmax_loc)
         
      time_step += 1

      file = File('pw'+ str(int(time_step)) +'.pvd')
      file << pw1
      file = File('pa'+ str(int(time_step)) +'.pvd')
      file << pa1
      file = File('pb'+ str(int(time_step)) +'.pvd')
      file << pb1
      file = File('pv'+ str(int(time_step)) +'.pvd')
      file << pv1

      pw_n.assign(pw1)
      pa_n.assign(pa1)
      pb_n.assign(pb1)
      pv_n.assign(pv1)
   
      fFile = HDF5File(MPI.comm_world,"pn"+str(int(pmax))+".h5","w")
      fFile.write(pw1,"/f")
      fFile.close()

t1=time.time()
print('time_taken', t1-t0)














