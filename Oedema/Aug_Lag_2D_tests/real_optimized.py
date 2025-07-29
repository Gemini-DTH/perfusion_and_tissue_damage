#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fenics
from fenics import *
import numpy as np 
import matplotlib.pyplot as plt
import dolfin
from mshr import *
from mshr.cpp import Sphere
import copy
import finite_element_fcts as fe_mod

set_log_active(False)

#Parameters for the Poroelastic Model
p_element_degree = 1
u_element_degree = 2
p_baseline_val = 1330#normal interstitial pressure
p_blood_val = Constant(4389)#feeding blood pressure 
vent_radius = 0.02
skull_radius = 0.07
infarct_radius = 0.02
nu = 0.35#poisson ratio
G = 216.3*2*1.37 #shear modulus
k_blood = 1E-10#blood conductivity
viscosity_blood = 3.6E-3#blood viscosity pa.s
Q_blood = 669 #blood relative compressibility
k_water = 3.6E-15#water conductivity
viscosity_water = 1E-3#water viscosity
Q_water = 3244#water relative compressibility
L_p = 3E-11#capillary wall permeability
tao = 0.93#reflection coefficient
os_p = 2445 #osmotic pressure
E_cap = 864.5#capillary wall stiffness
R_cap = 5E-6#capillary radius
n_b = 0.03 #blood volumn fraction
u_vent = Constant((0.0, 0.0)) 
mu = Constant(2*G)   	
lmbda = Constant(2*G/(1.0 - 2.0*nu))

#Create mesh and Geometry
center = dolfin.Point(0, 0)
center_infarct = dolfin.Point(0.045, 0)
vent_centre = dolfin.Point(0, 0)
infarct= Circle(center_infarct, infarct_radius, 200)
sphere = Circle(center, skull_radius) - infarct - Rectangle(Point(-0.023, -0.023),Point(0.023, 0.023)) #- Circle(vent_centre, 0.023, 200) #
domain = Circle(center, skull_radius) - Rectangle(Point(-0.023, -0.023),Point(0.023, 0.023))  # - Circle(vent_centre, 0.023, 200) #
domain.set_subdomain(2, sphere)
domain.set_subdomain(1, infarct)
mesh = generate_mesh(domain, 60)
markers = MeshFunction('size_t', mesh, 2, mesh.domains())
dx = Measure('dx', domain=mesh, subdomain_data=markers)
plot(mesh)
plt.show()
#Define Boundaries
TOL = 1E-5
class BoundarySkull(SubDomain):
        
        def inside(self, x, on_boundary):
            dx = x[0] 
            dy = x[1] 
            r = sqrt(dx*dx + dy*dy)
            return on_boundary and r > (skull_radius-TOL)

class BoundaryVent(SubDomain):
        
        def inside(self, x, on_boundary):
            dx = x[0] 
            dy = x[1] 
            r = sqrt(dx*dx + dy*dy)
            return on_boundary and abs(x[0]) < 0.023 + TOL and abs(x[1]) < 0.023 + TOL #  r < 0.025 + TOL #

skull_boundary = BoundarySkull()
vent_boundary = BoundaryVent() 
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
skull_boundary.mark(boundaries, 2)
vent_boundary.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
bbtree = mesh.bounding_box_tree()

#Collision Vertex and Colliding Cell
mesh1 = Mesh(mesh)
bbtreecom = mesh1.bounding_box_tree()
bbtreecom.build(mesh1)

class CollisionCounterCompare: #To OPTIMIZE (the collisions in the unddeformed cells are counted every time so this can be optimized. 
    _cpp_code = """
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    #include <pybind11/stl.h>
    namespace py = pybind11;
    
    #include <dolfin/mesh/Mesh.h>
    #include <dolfin/mesh/Vertex.h>
    #include <dolfin/mesh/MeshFunction.h>
    #include <dolfin/mesh/Cell.h>
    #include <dolfin/mesh/Vertex.h>
    #include <dolfin/geometry/BoundingBoxTree.h>
    #include <dolfin/geometry/Point.h>
    
    #include <iostream>
    #include <algorithm>
    #include <iterator>
    #include <set>
    #include <vector>
    using namespace dolfin;
    
    std::vector<int>
    compute_collisions(std::shared_ptr<const Mesh> mesh, std::shared_ptr<const Mesh> mesh1)
    {
      std::vector<unsigned int> colliding_cells;
      std::vector<unsigned int> colliding_cells_post;
      //BoundingBoxTree bbtree;
      //bbtree.build(*mesh);

      int num_vertices;
      num_vertices = mesh->num_vertices();
      
      //std::vector<int> collisions(num_vertices);
      //std::vector<int> collisions_post(num_vertices);
      std::vector<int> collisions_vertices(num_vertices);
       
      int i = 0;
      //std::set<int> ss;
  
      for (VertexIterator v(*mesh), v1(*mesh1); !v.end(); ++v, ++v1)
      {
        
        colliding_cells = mesh->bounding_box_tree()->compute_entity_collisions(v->point());
        colliding_cells_post = mesh1->bounding_box_tree()->compute_entity_collisions(v1->point());
        //collisions[i] = colliding_cells.size(); 
        //collisions_post[i] = colliding_cells_post.size();
        
        std::sort(colliding_cells.begin(), colliding_cells.end());
        std::sort(colliding_cells_post.begin(), colliding_cells_post.end());
        std::vector<int> ss;
        std::set_symmetric_difference(
        colliding_cells.begin(), colliding_cells.end(),
        colliding_cells_post.begin(), colliding_cells_post.end(),   
        std::back_inserter(ss));

        if (ss.size()!=0)
        {
            auto first = ss.begin(); 
            collisions_vertices[i] = *first;  
        } else {
            collisions_vertices[i] = -10;
        }  
        ++i;
        ss.clear();
      }
      return collisions_vertices;
    }
    
    PYBIND11_MODULE(SIGNATURE, m)
    {
      m.def("compute_collisions", &compute_collisions);
    }
    """
    _cpp_object = fenics.compile_cpp_code(_cpp_code)

    def __init__(self):
        pass

    @classmethod
    def compute_collisions(cls, mesh, mesh1):
        return np.array(cls._cpp_object.compute_collisions(mesh, mesh1))

#Define Linear Elasticity
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lmbda*div(u)*Identity(g) + mu*epsilon(u)

#Define BCs and variational problem
#W = VectorFunctionSpace(mesh, "CG", 2)
#Q = FunctionSpace(mesh, "CG", 1)
#V = W * Q

#element = MixedElement([cgpw, cgpb, cgpa, cgpv]) 
#V = FunctionSpace(mesh, element)

Vp = FunctionSpace(mesh, 'CG', p_element_degree)
Vu = VectorFunctionSpace(mesh, 'CG', u_element_degree)
bcpvz = DirichletBC(Vp, Constant(p_baseline_val), boundaries, 1)
bcpsz = DirichletBC(Vp, Constant(p_baseline_val), boundaries, 2)
bcusz = DirichletBC(Vu, u_vent, skull_boundary)
bcs_fluid = [bcpvz, bcpsz]
bcs_solid = [bcusz] 

p = Function(Vp)
u = Function(Vu)  
u_n = Function(Vu) 
p_n = Function(Vp)
vu = TestFunction(Vu)
vp = TestFunction(Vp)
g = u.geometric_dimension()
   
#Governing Equations
LHS = (k_water/viscosity_water)*dot(grad(p), grad(vp))*dx
RHS = n_b*(L_p/R_cap)* (p_blood_val - p - tao*os_p)*(1+(exp(1 + (100*(1 + (p_baseline_val - p)/E_cap)))-exp(-1 - (100*(1 + (p_baseline_val - p)/E_cap))))/(exp(1 + (100*(1 + (p_baseline_val - p)/E_cap))) + exp(-1 - (100*(1 + (p_baseline_val - p)/E_cap)))))*vp*dx #(1)
F = LHS - RHS

# Define Distance Function
Vk_function = FunctionSpace(mesh, 'CG', 1)
vertex_distance_to_boundary_function = Function(Vk_function) 
vertex_distance_to_boundary_functiondot = Function(Vk_function) 
u_n = Function(Vu)

# Compute solution and Computational parameters to initialize
n = FacetNormal(mesh)
u_magnitude_max = 1.0
distance_max = 1.0 
penaltycoeff = 50
step = 0

#Simulation and Move Mesh    
J = derivative(F, p)
problem=NonlinearVariationalProblem(F,p,bcs_fluid,J)
solver=NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'mumps'
solver.solve()
pmax = p.vector().get_local().max()
print('pmax:', pmax)
b = plot(p)
plt.colorbar(b)
plot(mesh)
plt.show()

#gradp = project(grad(p), Vu)

while u_magnitude_max >= 5E-4:#or step<6:

   u = TrialFunction(Vu)
   RHS_solid = - dot(grad(p),vu)*dx
   LHS_solid = inner(sigma(u + u_n), epsilon(vu))*dx + penaltycoeff*(vertex_distance_to_boundary_function)*dot(n,vu)*ds     #
   F_solid = LHS_solid - RHS_solid
   RHS = rhs(F_solid) 
   LHS = lhs(F_solid) 
   u = Function(Vu)
   
   lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, True #bicgstab
   u = fe_mod.solve_lin_sys(Vu,LHS,RHS,bcs_solid,lin_solver,precond,rtol,mon_conv,init_sol) 
   #J = derivative(F_solid, u)
   #problem=NonlinearVariationalProblem(F_solid,u,bcs_solid,J)
   #solver=NonlinearVariationalSolver(problem)
   #prm = solver.parameters
   #prm['nonlinear_solver'] = 'newton'
   #prm['newton_solver']['linear_solver'] = 'mumps'
   #solver.solve()

   #problem = Problem(J, F, bcs_solid)
   #custom_solver = CustomSolver()
   #custom_solver.solve(problem, u.vector())
   
   u_magnitude = sqrt(dot(u, u))
   u_magnitude = project(u_magnitude, Vp)
   u_magnitude_max=u_magnitude.vector().get_local().max()
   
   if step > 1 and u_magnitude_max > 0.001: 
      u = project(u*0.1, Vu)    
   
   ALE.move(mesh, u)
   print('maximum displacement in the step = ', u_magnitude_max)   
   print('mesh move!')
   u_n.assign(u + u_n) #
   
   u_magnitude = sqrt(dot(u, u))
   u_magnitude = project(u_magnitude, Vp)
   b = plot(u_magnitude)
   plt.colorbar(b)
   plot(mesh)
   plt.show()
   #u_n = Function(Vu) 


   #Contact Vertex and Colliding Cell ( I am using a lot of numpy functions to obtain the ID of colliding vertices. Maybe it is possible to incorporate this into the C++ so it can be simplier.                                                  
   bbtree = mesh.bounding_box_tree()
   bbtree.build(mesh)
   collisions_vertices_idx = CollisionCounterCompare.compute_collisions(mesh, mesh1)
   collisions_vertices = copy.copy(collisions_vertices_idx) 
   collisions_vertices_idx[np.where(collisions_vertices_idx >= 0)] = 1
   c = len(collisions_vertices_idx)
   contact_point = collisions_vertices_idx*range(c)
   contact_point = contact_point[contact_point > 0]
   if collisions_vertices[0] >= 0:
      contact_point = np.insert(contact_point,0,0)

   #Boundary Mesh and Mapping
   bmesh = BoundaryMesh(mesh, "exterior")
   bdim = bmesh.topology().dim()
   mapping = bmesh.entity_map(0)
   boundary_backto_directory = []
   boundary_backto_directoryb = []
   for v in vertices(bmesh):
       boundary_backto_directory.append(mapping[v])
       boundary_backto_directoryb.append(v.index()) 
   
   #Contact Boundary Vertex
   boundary_verticenum = [mapping[v] 
                             for v in vertices(bmesh)]
  
   contact_vertices = list(set(boundary_verticenum) & set(contact_point)) 

   Vk_function = FunctionSpace(mesh, 'CG', 1)
   V2D= vertex_to_dof_map(Vk_function)
   #vertex_distance_to_boundary_function = Function(Vk_function) 

   print(collisions_vertices)
   print(len(contact_vertices))
   
   #Find Closest Boundary to a Penetrating Vertex
   for v_idx in contact_vertices:
       v = Vertex(mesh, v_idx) 
       cell_idx = collisions_vertices[v_idx]
       contain_cell = Cell(mesh, cell_idx)
       cell_vertex_id = contain_cell.entities(0) 

       #Boundary Detect
       while not (list(set(cell_vertex_id) & set(contact_vertices))):  # OPTIMIZE (I look at whether there is any vertex of the cells on the boundary. If not, I keep finding the neighbouring cells, many cells are counted repeatedly. Maybe there is a better way to do this in Dolfinx 

          neighborhood = []

          for cell_v_id in cell_vertex_id:
              cell_v_id = int(cell_v_id)
              vidx = Vertex(mesh, cell_v_id)
              for cell_nei in vidx.entities(mesh.topology().dim()):                
                  cell_neibh = Cell(mesh, cell_nei)
                  boundary_detect = list(set(cell_neibh.entities(0)))
                  for ix in boundary_detect:
                      neighborhood.append(ix)

          cell_vertex_id = list(set(neighborhood))  
       
       #Boundary Vertex to Create BoundaryMesh ( I create submeshes for each vertex to measure the distance, maybe we can do this in parallel or by using some other function available now.
       sub_vertex = (list(set(cell_vertex_id) & set(contact_vertices))) 
       boundaries_sub = MeshFunction("size_t", bmesh, bmesh.topology().dim())
       boundaries_sub.set_all(0)    
       for v_sub_idx in sub_vertex:
           v_sub = boundary_backto_directoryb[boundary_backto_directory.index(v_sub_idx)]       
           v_sub = Vertex(bmesh, v_sub)
           for cell_sub in cells(v_sub): 
               boundaries_sub[cell_sub] = 1000

       submesh_sub = SubMesh(bmesh,boundaries_sub,1000) 

       bbtree_sub = BoundingBoxTree()
       bbtree_sub.build(submesh_sub)
       _, distance = bbtree_sub.compute_closest_entity(v.point())
       vertex_distance_to_boundary_function.vector()[V2D[v_idx]] += 1#distance 

   # the Distance function
   distance_max = vertex_distance_to_boundary_function.vector().get_local().max()
   #vertex_distance_to_boundary_function.vector()[:] = vertex_distance_to_boundary_function.vector()/Constant(distance_max)

   print('maximum displacement in the step = ', u_magnitude_max, 'maximum penetration depth = ',distance_max)
   step += 1

#Plot Results
#c = plot(vertex_distance_to_boundary_function)
#plt.colorbar(c)
#plt.show()
u_magnitude = sqrt(dot(u_n, u_n))
u_magnitude = project(u_magnitude, Vp)
b = plot(u_magnitude)
plt.colorbar(b)
plot(mesh)
plt.show()

