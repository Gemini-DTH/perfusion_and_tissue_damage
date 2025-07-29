#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
CX
"""
from dolfin import *
import numpy as np
import meshio
import matplotlib.pyplot as plt
import timeit
import os 

newpath = r'/home/chenxi/桌面/OSMO/'+ str(ids)  
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
blood_p = blood_pressure


def fun (x): 
    if 0 <= x <= 15:
        return maxcon/(1+(np.exp(15*(7.5 -x)/15))) #*(np.exp(0.25*(x))- np.exp(-0.25*(x)))/(np.exp(0.25*(x)) + np.exp(-0.25*(x))) #/(np.exp(0.15*(7.5))- np.exp(-0.15*(7.5)))*(np.exp(0.15*(7.5)) + np.exp(-0.15*(7.5)))) #    () -1800+2*200*9/(1+(np.exp(7.5*(-x)/15)))#
    elif 15 < x:
        return maxcon*np.exp(-0.01424*(x-15))  # 350 11.2 160.7142 tao & osp justify # 200  #-0.0244

concentration = np.vectorize(fun)

t = np.linspace(0, 245, 2450)
osmo = concentration(t)
plt.plot(t, osmo)
plt.scatter([0, 15, 60],[0, 200*9, 600], c='r') #948.21
#plt.show()
print(osmo[50])

dt = 60
num_step = 240
#########PERFUSION
mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 3)
mvc_b = MeshValueCollection("size_t", mesh, 2)
  
with XDMFFile("md.xdmf") as infile:
    infile.read(mvc, "name_to_read")
md = cpp.mesh.MeshFunctionSizet(mesh, mvc)

with XDMFFile("mf.xdmf") as infile:
    infile.read(mvc_b, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc_b)

class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "hypre")
        PETScOptions.set("pc_hypre_type", "euclid")

        self.linear_solver().set_from_options()

Vpe = FunctionSpace(mesh, "Lagrange", 1)   
dx = Measure("dx", domain=mesh, subdomain_data = md)#, subdomain_id=50)
bcsp = []
for i in range(6):
   a = i + 21
   #if 'Dirichlet' in boundary_conditions[i]:
   bc = DirichletBC(Vpe, Constant(1), mf, a)
   bcsp.append(bc)
bcv = DirichletBC(Vpe, Constant(0), mf, 2)
bcsp.append(bcv)
bc = DirichletBC(Vpe, Constant(1), mf, 32)
bcsp.append(bc)
bc = DirichletBC(Vpe, Constant(1), mf, 31)
bcsp.append(bc)

pe = TrialFunction(Vpe)
ve = TestFunction(Vpe)
f = Constant(0.0)

LHS = inner(grad(pe), grad(ve))*dx
RHS = f*ve*dx 	
pe = Function(Vpe)

solve(LHS == RHS, pe, bcsp, solver_parameters={'linear_solver':'bicgstab'})

#file = File('pe.pvd')
#file << pe

Ve = VectorFunctionSpace(mesh, "Lagrange", 1)
Ve_DG = VectorFunctionSpace(mesh, "DG", 0)
Vdir = FunctionSpace(mesh, "DG", 0) # function space to store major direction
e = project(-grad(pe),Ve, solver_type='bicgstab')
e = interpolate(e,Ve_DG)
e_array = e.vector().get_local()

for i in range(int(len(e_array)/3)):
    e_array[i*3:(i+1)*3] = e_array[i*3:(i+1)*3]/np.linalg.norm(e_array[i*3:(i+1)*3])
e.vector().set_local(e_array)

#file = File('e.pvd')
#file << e

def comp_transf_mat(e0,e1):
    v = np.cross(e0,e1)
    s = np.linalg.norm(v) # sine
    c = np.dot(e0,e1) # cosine
    I = np.identity(3) # identity matrix
    u = v/s
    ux = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    T = c*I + s*ux + (1-c)*np.tensordot(u,u,axes=0)

    return T

K_space = TensorFunctionSpace(mesh, "DG", 0)
e_ref = np.array([0,0,1])
K1_form = [[0,0,0],[0,0,0],[0,0,1]]
K1 = Function(K_space)    
K1_array = K1.vector().get_local()
e_loc_array = e.vector().get_local()

for i in range(mesh.num_cells()):
    e1 = e_loc_array[i*3:i*3+3]
    T = comp_transf_mat(e_ref,e1)                                 
        
    # rotate local permeability tensor
    K1_loc = np.reshape(np.matmul(np.matmul(T,K1_form),T.transpose()),9 ) 
        
    # set local permeability tensor
    K1_array[i*9:i*9+9] = K1_loc
   
tolerance = 1e-9
mask = abs(K1_array) < tolerance
K1_array[mask] = 0
K1.vector().set_local(K1_array)

#file = File('K1.pvd')
#file << K1

Ve_te = TensorFunctionSpace(mesh, "DG", 0)
K = project(K1,Ve_te, solver_type='bicgstab')
K = interpolate(K,Ve_te)
print('k')

x = mesh.coordinates(); x[:,:] *= 0.001
#Constant 
k_blood =  1.5408e-14 # ##ten times!!!!! 1E-14 #*1e-5   #1e-15 # #1e-12  delta 50   1e-13 200 (wahbi)   7大10小      
k_blood_a = 4.4424e-12 
k_blood_v = k_blood_a*2 
viscosity_blood = 3.6E-3 
k_water =  3.6e-15*Kr #kwater
viscosity_water = 1E-3 
tracg = 1.326e-6  
tracw = 5.22e-7  
trcvg = 4.641e-6
trcvw = 1.828e-6
p_baseline_val = 666.5 #1330 #330
L_p = 3e-11*Lpr #3e-11 # !!!!!!!!!!11 #3e-10 # 
tao = 0.35 #.13  # changed
os_p = 2445 #+ 785
E_cap =  864.5 
R_cap = 5E-6 
n_b = 0.03 
compressi = 1/Cr
print(L_p)
print('kwater=',k_water)
print('tao',tao)
print('lp',L_p)
blockage = [21,22,23]
print(blockage)
print(compressi)
#Define variational problem 
element_type = mesh.ufl_cell()
cgpw = FiniteElement('CG', element_type, 1)
cgpwt = FiniteElement('Real', element_type, 0)
cgpb = FiniteElement('CG', element_type, 1)
cgpa = FiniteElement('CG', element_type, 1)
cgpv = FiniteElement('CG', element_type, 1)
element = MixedElement([cgpw, cgpb, cgpa, cgpv]) 
V = FunctionSpace(mesh, element)
fcts = Function(V)
vpw, vpb, vpa, vpv = TestFunction(V)
pw, pb, pa, pv= split(fcts)

# Define boundary condition

bcsp = []
for i in range(6):
   a = i + 21
   bc = DirichletBC(V.sub(2), Constant(blood_p), mf, a)
   bcsp.append(bc)
bc = DirichletBC(V.sub(2), Constant(blood_p), mf, 32)
bcsp.append(bc)
bc = DirichletBC(V.sub(2), Constant(blood_p), mf, 31)
bcsp.append(bc)

for i in range(6):
   a = i + 21
   bc = DirichletBC(V.sub(3), Constant(2000), mf, a)
   bcsp.append(bc)
bc = DirichletBC(V.sub(3), Constant(2000), mf, 32)
bcsp.append(bc)
bc = DirichletBC(V.sub(3), Constant(2000), mf, 31)
bcsp.append(bc)


for i in range(6):
   a = i + 21
   bc = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, a)
   bcsp.append(bc)
bcv = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, 2)
bcsp.append(bcv)
bc = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, 32)
bcsp.append(bc)
bc = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, 31)
bcsp.append(bc)

#Governing Equation 
Vk = FunctionSpace(mesh, "Lagrange", 1)  
 
LHSw = (k_water/viscosity_water)*dot(grad(pw), grad(vpw))*dx 

LHSb = (k_blood/viscosity_blood)*dot(grad(pb), grad(vpb))*dx 

RHSb =  trcvw*(pb-pv)*vpb*dx(11) - tracg*(pa-pb)*vpb*dx(12) + trcvg*(pb-pv)*vpb*dx(12) - tracw*(pa-pb)*vpb*dx(11) 

LHSa = (k_blood_a/viscosity_blood)*dot(K*grad(pa), grad(vpa))*dx 

RHSa = tracg*(pa-pb)*vpa*dx(12) + tracw*(pa-pb)*vpa*dx(11) 

LHSv = (k_blood_v/viscosity_blood)*dot(K*grad(pv), grad(vpv))*dx 

RHSv = - trcvg*(pb-pv)*vpv*dx(12) - trcvw*(pb-pv)*vpv*dx(11)
		
F =   LHSb + RHSb + LHSw + RHSa + LHSa + LHSv + RHSv 
 
J = derivative(F, fcts)
problem = Problem(J, F, bcsp)
custom_solver = CustomSolver()
custom_solver.solve(problem, fcts.vector())
(pw, pb, pa, pv) = fcts.split(True)

perfusion = project((pa-pb),Vk, solver_type='bicgstab', preconditioner_type='amg')



file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) + '/pb.pvd')
file << pb
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pw.pvd')
file << pw
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pa.pvd')
file << pa
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pv.pvd')
file << pv
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/perfu.pvd') #xdmf
file << perfusion

bcsp = []
for i in range(6):
   a = i + 21
   if a not in blockage:
      bc = DirichletBC(V.sub(2), Constant(blood_p), mf, a)
      bcsp.append(bc)
   bcsp.append(bc)
bc = DirichletBC(V.sub(2), Constant(blood_p), mf, 32)
bcsp.append(bc)
bc = DirichletBC(V.sub(2), Constant(blood_p), mf, 31)
bcsp.append(bc)

for i in range(6):
   a = i + 21
   bc = DirichletBC(V.sub(3), Constant(2000), mf, a)
   bcsp.append(bc)
bc = DirichletBC(V.sub(3), Constant(2000), mf, 32)
bcsp.append(bc)
bc = DirichletBC(V.sub(3), Constant(2000), mf, 31)
bcsp.append(bc)

for i in range(6):
   a = i + 21
   bc = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, a)
   bcsp.append(bc)
bcv = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, 2)
bcsp.append(bcv)
bc = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, 32)
bcsp.append(bc)
bc = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, 31)
bcsp.append(bc)


J = derivative(F, fcts)
problem = Problem(J, F, bcsp)
custom_solver = CustomSolver()
custom_solver.solve(problem, fcts.vector())
(pw, pb, pa, pv) = fcts.split(True)

perfusion1 = project((pa-pb),Vk, solver_type='bicgstab', preconditioner_type='amg')

file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pb1.pvd')
file << pb
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pw1.pvd')
file << pw
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pa1.pvd')
file << pa
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pv1.pvd')
file << pv
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/perfu1.pvd') #xdmf
file << perfusion1

pwo_0 = interpolate(pw, Vk)
pao_0 = interpolate(pa, Vk)
pbo_0 = interpolate(pb, Vk)
pvo_0 = interpolate(pv, Vk)

infarctflow = project((perfusion - perfusion1)/perfusion, Vk, solver_type='bicgstab', preconditioner_type='amg')
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/infarctflow.pvd') 
file << infarctflow

subgrey = list(md.array() == 11)
#print(subgrey)

class RegionOfInterestg(SubDomain):
    def set_threshold(self,threshold):
        self.threshold = threshold
    def inside(self,x,on_boundary):
       return (infarctflow(x) >= self.threshold) #and x in [md.array() == 12] 

class RegionOfInterestw(SubDomain):
    def set_threshold(self,threshold):
        self.threshold = threshold
    def inside(self,x,on_boundary):
       return (infarctflow(x) >= self.threshold) #and x in [md.array() == 11]

ds = Measure('ds', domain=mesh, subdomain_data=mf)
v_expr = Expression('1', degree=1)
v_exe = interpolate(v_expr, Vk)
#print('blokage id:',blockage)
regiong = RegionOfInterestg()
regiong.set_threshold(0.7)
regiong.mark(md, 50) 
regionw = RegionOfInterestw()
regionw.set_threshold(0.7)
regionw.mark(md, 49)  
submesh_regiong = SubMesh(mesh, regiong)
vmap = submesh_regiong.data().array('parent_vertex_indices', 0)
cmap = submesh_regiong.data().array('parent_cell_indices', 3)

for submesh_cell in cells(submesh_regiong): # loop for copying the names of the subdomains
    parent_cell = cmap[submesh_cell.index()]
    if subgrey[parent_cell] == True: #.index() 
       md.array()[parent_cell] = 50
    else:
       md.array()[parent_cell] = 49

dx1 = Measure("dx", subdomain_data = md, subdomain_id=50)
dx2 = Measure("dx", subdomain_data = md, subdomain_id=49)
vol = assemble(v_exe*dx1) + assemble(v_exe*dx2)
vol_total = assemble(v_exe*dx)
submesh_regionw = SubMesh(mesh, md, 50)
submesh_regiong = SubMesh(mesh, md, 49)

#xdmf_filename = XDMFFile(MPI.comm_world, "RegionOfInterestg.xdmf")
#dmf_filename.write(submesh_regiong)
#xdmf_filename = XDMFFile(MPI.comm_world, "RegionOfInterestw.xdmf")
#xdmf_filename.write(submesh_regionw)
print(vol,vol_total)
volgw = assemble(v_exe*dx(11)) + assemble(v_exe*dx(12))
print('111111111',volgw)

#Define variational problem 
V = FunctionSpace(mesh, element)

fctss = Function(V)
vpwo, vpbo, vpao, vpvo = TestFunction(V)
pwo, pbo, pao, pvo= split(fctss)
Vv = FunctionSpace(mesh, 'CG', 1)
Vk = FunctionSpace(mesh, 'CG', 1)

#initial pressure
fctss = Function(V)
vpwoz,  vpboz, vpaoz, vpvoz = TestFunction(V)
pwoz, pboz, paoz, pvoz = split(fctss)

bcsp = []
for i in range(6):
   a = i + 21
   bc = DirichletBC(V.sub(2), Constant(blood_p), mf, a)

   bcsp.append(bc)
bc = DirichletBC(V.sub(2), Constant(blood_p), mf, 32)
bcsp.append(bc)
bc = DirichletBC(V.sub(2), Constant(blood_p), mf, 31)
bcsp.append(bc)

for i in range(6):
   a = i + 21
   bc = DirichletBC(V.sub(3), Constant(2000), mf, a)
   bcsp.append(bc)
bc = DirichletBC(V.sub(3), Constant(2000), mf, 32)
bcsp.append(bc)
bc = DirichletBC(V.sub(3), Constant(2000), mf, 31)
bcsp.append(bc)

for i in range(6):
   a = i + 21
   bc = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, a)
   bcsp.append(bc)
bcv = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, 2)
bcsp.append(bcv)
bc = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, 32)
bcsp.append(bc)
bc = DirichletBC(V.sub(0), Constant(p_baseline_val), mf, 31)
bcsp.append(bc)

i = 0
pmax = 11
while pmax >= 10: 

   fctss = Function(V)
   vpwo,  vpbo, vpao, vpvo = TestFunction(V)
   pwo, pbo, pao, pvo = split(fctss)

   fo = Constant(1) #0.5*(1+(exp((1 + (p_baseline_val - pwo)/E_cap))-exp(-(1 + (p_baseline_val - pwo)/E_cap)))/(exp((1 + (p_baseline_val - pwo)/E_cap)) + exp(-(1 + (p_baseline_val - pwo)/E_cap))))

   LHSw = (pwo - pwo_0)*vpwo/300/compressi/3244*dx + (k_water/viscosity_water)*dot(grad(pwo), grad(vpwo))*dx #
   RHSw =  -2*n_b*fo*(L_p/R_cap)*(pbo - pwo - tao*os_p)*vpwo*dx(49) -2*n_b*fo*(L_p/R_cap)*(pbo - pwo - tao*os_p )*vpwo*dx(50)

   LHSb = (pbo-pbo_0)*vpbo/300/compressi/669*dx + (k_blood/viscosity_blood)*dot(grad(pbo), grad(vpbo))*dx  
   RHSb =  fo*trcvw*(pbo-pvo)*vpbo*dx(11) + fo*trcvg*(pbo-pvo)*vpbo*dx(12) + fo*trcvw*(pbo-pvo)*vpbo*dx(50) + fo*trcvg*(pbo-pvo)*vpbo*dx(49)- fo*tracw*(pao-pbo)*vpbo*dx(11) - fo*tracg*(pao-pbo)*vpbo*dx(12) - fo*tracw*(pao-pbo)*vpbo*dx(50) - fo*tracg*(pao-pbo)*vpbo*dx(49) + 2*n_b*fo*(L_p/R_cap)*(pbo - pwo - tao*os_p)*vpbo*dx(50) + 2*n_b*fo*(L_p/R_cap)*(pbo - pwo - tao*os_p)*vpbo*dx(49)

   LHSa =  (pao-pao_0)*vpao/300/compressi/669*dx + (k_blood_a/viscosity_blood)*dot(K*grad(pao), grad(vpao))*dx #
   RHSa = fo*tracw*(pao-pbo)*vpao*dx(11) + fo*tracg*(pao-pbo)*vpao*dx(12) + fo*tracw*(pao-pbo)*vpao*dx(50) + fo*tracg*(pao-pbo)*vpao*dx(49)

   LHSv =  (pvo-pvo_0)*vpvo/300/compressi/669*dx + (k_blood_v/viscosity_blood)*dot(K*grad(pvo), grad(vpvo))*dx # 
   RHSv = - fo*trcvw*(pbo-pvo)*vpvo*dx(11) - fo*trcvg*(pbo-pvo)*vpvo*dx(12) - fo*trcvg*(pbo-pvo)*vpvo*dx(49)  - fo*trcvw*(pbo-pvo)*vpvo*dx(50) 
   F =  LHSw +  RHSw + RHSa + LHSa + LHSv + RHSv  + LHSb + RHSb 

   J = derivative(F, fctss)
   problem = Problem(J, F, bcsp)
   custom_solver = CustomSolver()
   custom_solver.solve(problem, fctss.vector())
   (pwo, pbo, pao, pvo) = fctss.split(True)
   pmax = project((pwo-pwo_0),Vk, solver_type='bicgstab', preconditioner_type='amg')
   pmax = pmax.vector().get_local().max()
   p_ma = pwo_0.vector().get_local().max()
   p_max = pwo.vector().get_local().max()
   pwo_0.assign(pwo)
   pbo_0.assign(pbo)
   pao_0.assign(pao)
   pvo_0.assign(pvo)
   p_mao = pwo_0.vector().get_local().max()
   print(p_max, pmax, p_ma, p_mao)
   
   #u = TrialFunction(Vu)
   #a = inner(sigma(u), epsilon(vu))*dx 
   #L = - dot(grad(pwo),vu)*dx - inner(sigma(u_n), epsilon(vu))*dx #- dot(grad(pb),vu)*dx #+ dot(grad(p_nw),vu)*dx + dot(grad(p_nb),vu)*dx# + dot(AA, vu)*ds #1000*   
   #u = Function(Vu)
   #solve(a == L, u, bcs_solid,  solver_parameters={'linear_solver': 'bicgstab','preconditioner': 'amg'})
   #u_magnitude = sqrt(dot(u, u))
   #u_magnitude = project(u_magnitude, Vv)
   #u_magnitude_max=u_magnitude.vector().get_local().max()
   #print(u_magnitude_max)
   p_magnitude = project(pwo, Vv)
   p_magnitude_max=p_magnitude.vector().get_local().max()
   print(p_magnitude_max)
   #u_n1 = interpolate(u, Vu) 
   #u_n.assign(u_n1 + u_n)
   #ALE.move(mesh, u_n1)
   i += 1
   
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pbo.pvd')
file << pbo
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pwo.pvd')
file << pwo
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pao.pvd')
file << pao
file = File(r'/home/chenxi/桌面/OSMO/'+ str(ids) +  '/pvo.pvd')
file << pvo
   
pwo_0 = interpolate(pwo, Vk)
pao_0 = interpolate(pao, Vk)
pbo_0 = interpolate(pbo, Vk)
pvo_0 = interpolate(pvo, Vk)
File('saved_pwo.xml') << pwo
File('saved_pao.xml') << pao
File('saved_pbo.xml') << pbo
File('saved_pvo.xml') << pvo
pwo_0 = Function(Vk, 'saved_pwo.xml')
pao_0 = Function(Vk, 'saved_pao.xml')
pvo_0 = Function(Vk, 'saved_pvo.xml')
pbo_0 = Function(Vk, 'saved_pbo.xml')

#
i =0
t = 0   
pressure = np.linspace(0,240,55)
pressure[i] = p_max
while i <= 47:
 
   t += 50 
   i += 1
   osmox = osmo[t]
   print(osmox)
   fctss = Function(V)
   vpwo,  vpbo, vpao, vpvo = TestFunction(V)
   pwo, pbo, pao, pvo= split(fctss)

   fo = Constant(1)#0.5*(1+(exp((1 + (p_baseline_val - pwo)/E_cap))-exp(-(1 + (p_baseline_val - pwo)/E_cap)))/(exp((1 + (p_baseline_val - pwo)/E_cap)) + exp(-(1 + (p_baseline_val - pwo)/E_cap))))

   LHSw = (pwo - pwo_0)*vpwo/300/compressi/3244*dx + (k_water/viscosity_water)*dot(grad(pwo), grad(vpwo))*dx #

   RHSw =  -2*n_b*fo*(L_p/R_cap)*(pbo - pwo - tao*os_p - osmox)*vpwo*dx(49) -2*n_b*fo*(L_p/R_cap)*(pbo - pwo - tao*os_p - osmox)*vpwo*dx(50)

   LHSb = (pbo-pbo_0)*vpbo/300/compressi/669*dx + (k_blood/viscosity_blood)*dot(grad(pbo), grad(vpbo))*dx # 

   RHSb =  fo*trcvw*(pbo-pvo)*vpbo*dx(11) + fo*trcvg*(pbo-pvo)*vpbo*dx(12) + fo*trcvw*(pbo-pvo)*vpbo*dx(50) + fo*trcvg*(pbo-pvo)*vpbo*dx(49)- fo*tracw*(pao-pbo)*vpbo*dx(11) - fo*tracg*(pao-pbo)*vpbo*dx(12) - fo*tracw*(pao-pbo)*vpbo*dx(50) - fo*tracg*(pao-pbo)*vpbo*dx(49) + 2*n_b*fo*(L_p/R_cap)*(pbo - pwo - tao*os_p - osmox)*vpbo*dx(50) + 2*n_b*fo*(L_p/R_cap)*(pbo - pwo - tao*os_p - osmox)*vpbo*dx(49)

   LHSa =  (pao-pao_0)*vpao/300/compressi/669*dx + (k_blood_a/viscosity_blood)*dot(K*grad(pao), grad(vpao))*dx #

   RHSa = fo*tracw*(pao-pbo)*vpao*dx(11) + fo*tracg*(pao-pbo)*vpao*dx(12) + fo*tracw*(pao-pbo)*vpao*dx(50) + fo*tracg*(pao-pbo)*vpao*dx(49)

   LHSv =  (pvo-pvo_0)*vpvo/300/compressi/669*dx + (k_blood_v/viscosity_blood)*dot(K*grad(pvo), grad(vpvo))*dx # 

   RHSv = - fo*trcvw*(pbo-pvo)*vpvo*dx(11) - fo*trcvg*(pbo-pvo)*vpvo*dx(12) - fo*trcvg*(pbo-pvo)*vpvo*dx(49)  - fo*trcvw*(pbo-pvo)*vpvo*dx(50) 
   F =  LHSw +  RHSw + RHSa + LHSa + LHSv + RHSv  + LHSb + RHSb 

   J = derivative(F, fctss)

   problem = Problem(J, F, bcsp)

   custom_solver = CustomSolver()
   custom_solver.solve(problem, fctss.vector())

   (pwo, pbo, pao, pvo) = fctss.split(True)
   pwo_0.assign(pwo)
   pbo_0.assign(pbo)
   pao_0.assign(pao)
   pvo_0.assign(pvo)
   p_max = pwo.vector().get_local().max()
   p_min = pwo.vector().get_local().max()
   p_maxb = pbo.vector().get_local().max()  
   if p_max == 1330: 
      pressure[i] = p_min
   else:
      pressure[i] = p_max
   print(p_max,i, osmo[t], p_maxb)
   print(pressure)
   p_magnitude = project(pwo, Vv)
   p_magnitude_max=p_magnitude.vector().get_local().max()

x = [0,5,10,15,20,25,35,45,60,120,180,240]
p = [26.79, 25.44, 23.36,20.91,18.75,17.68,16.77,16.96,18.29,21.07,22.72,24.20]
a = 133*np.ones(12)
p = a*p


