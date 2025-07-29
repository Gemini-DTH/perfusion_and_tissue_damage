from dolfin import *
import numpy as np
import time
import os 
import dolfin as df

def solve_lin_sys(Vp,LHS,RHS,BCs,lin_solver,precond,rtol,mon_conv,init_sol,**kwarg):
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Define functions for pressures
    fcts = Function(Vp)
    #vp = TestFunction(Vp)
    # define weak form
    problem = LinearVariationalProblem(LHS, RHS, fcts, BCs)
        
    # TODO: set up initialisation with first order:
    # https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/?show=1124#q1124
    
    # solver settings
    solver = LinearVariationalSolver(problem)
    prm = solver.parameters
    #prm.keys()
    prm['linear_solver'] = lin_solver
    if precond != False:
        prm['preconditioner'] = precond
    if rtol != False:
        PETScOptions.set('ksp_rtol', str(rtol))
        prm['krylov_solver']['relative_tolerance']=rtol
    prm['krylov_solver']["monitor_convergence"] = mon_conv
    prm['krylov_solver']["nonzero_initial_guess"] = init_sol
    
    # solve equation system
    #start = time.time()
    solver.solve()
    return fcts

def solve_lin_sys1(V,LHS,RHS,bcsp,lin_solver,precond,rtol,mon_conv,init_sol,**kwarg):
    comm = MPI.comm_world
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Define functions for pressures
    fctss = Function(V)
    #vp = TestFunction(Vp)
    # define weak form
    problem = LinearVariationalProblem(LHS, RHS, fctss, bcsp)
        
    # TODO: set up initialisation with first order:
    # https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/?show=1124#q1124
    
    # solver settings
    solver = LinearVariationalSolver(problem)
    prm = solver.parameters
    #prm.keys()
    prm['linear_solver'] = lin_solver
    if precond != False:
        prm['preconditioner'] = precond
    if rtol != False:
        PETScOptions.set('ksp_rtol', str(rtol))
        prm['krylov_solver']['relative_tolerance']=rtol
    prm['krylov_solver']["monitor_convergence"] = mon_conv
    prm['krylov_solver']["nonzero_initial_guess"] = init_sol
    
    # solve equation system
    #start = time.time()
    solver.solve()
    return fctss

def sub_mesh(mesh, region):
    submesh_region = SubMesh(mesh, region)
    return submesh_region
    
def tensor_generate(mesh, md, mf):
   Vpe = FunctionSpace(mesh, "Lagrange", 1)   
   dx = Measure("dx", domain=mesh, subdomain_data = md)
   bcsp = []
   for i in [1,2,3,4,5,6,8]:
      a = i #+ 21
      bc = DirichletBC(Vpe, Constant(1), mf, a)
      bcsp.append(bc)
   bcv = DirichletBC(Vpe, Constant(0), mf, 7)
   bcsp.append(bcv)

   pe = TrialFunction(Vpe)
   ve = TestFunction(Vpe)
   f = Constant(0.0)

   LHS = inner(grad(pe), grad(ve))*dx
   RHS = f*ve*dx 	
   pe = Function(Vpe)

   solve(LHS == RHS, pe, bcsp, solver_parameters={'linear_solver':'bicgstab'})

   Ve = VectorFunctionSpace(mesh, "Lagrange", 1)
   Ve_DG = VectorFunctionSpace(mesh, "DG", 0)
   Vdir = FunctionSpace(mesh, "DG", 0) 
   e = project(-grad(pe),Ve, solver_type='bicgstab')
   e = interpolate(e,Ve_DG)
   e_array = e.vector().get_local()

   for i in range(int(len(e_array)/3)):
       e_array[i*3:(i+1)*3] = e_array[i*3:(i+1)*3]/np.linalg.norm(e_array[i*3:(i+1)*3])
   e.vector().set_local(e_array)

   def comp_transf_mat(e0,e1):
       v = np.cross(e0,e1)
       s = np.linalg.norm(v) 
       c = np.dot(e0,e1) 
       I = np.identity(3) 
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

   Ve_te = TensorFunctionSpace(mesh, "DG", 0)
   K = project(K1,Ve_te, solver_type='bicgstab')
   K = interpolate(K,Ve_te)
   return K 
    

