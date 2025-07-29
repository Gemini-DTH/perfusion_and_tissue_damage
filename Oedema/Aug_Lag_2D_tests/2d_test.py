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
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

ADD_MODE = PETSc.InsertMode.ADD

set_log_active(False)

#Parameters for the Poroelastic Model
p_element_degree = 1
u_element_degree = 1
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
infarct= Circle(center_infarct, infarct_radius)
sphere = Circle(center, skull_radius) - infarct - Rectangle(Point(-0.023, -0.023),Point(0.023, 0.023)) #- Circle(vent_centre, 0.023, 200) #
domain = Circle(center, skull_radius) - Rectangle(Point(0.02, -0.023),Point(0.023, 0.023))  # - Circle(vent_centre, 0.023, 200) #
domain.set_subdomain(2, sphere)
domain.set_subdomain(1, infarct)
mesh = generate_mesh(domain, 50)
markers = MeshFunction('size_t', mesh, 2, mesh.domains())
dx = Measure('dx', domain=mesh, subdomain_data=markers)
#plot(mesh)
#plt.show()
#Define Boundaries
TOL = 1E-3

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
            return on_boundary and (abs((x[0] - 0.0215) < 0.0015 + TOL) and abs(x[1]) < 0.023 + TOL) # abs(x[0]) < 0.023 + TOL and abs(x[1]) < 0.023 + TOL # r < 0.025 + TOL #

skull_boundary = BoundarySkull()
vent_boundary = BoundaryVent() 
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
skull_boundary.mark(boundaries, 2)
vent_boundary.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
bbtree = mesh.bounding_box_tree()
           
tag = 1
all_facets = np.array(list(facets(mesh)))
surfacefacets = all_facets[np.where(boundaries.array() == tag)[0]]

Penalty = 2000

def nodal_normal(bmesh):
   hashtable_nodal_normal = dict()
   for vertexindex in vertices(bmesh): 
      vertexfacets = list(filter(lambda x: mapping[vertexindex] in x.entities(0), surfacefacets))
      average_normal_vector = np.array([[facet.normal().x(),
                                         facet.normal().y()]
                                         for facet in vertexfacets]
                                         ).mean(axis=0)
      #print(vertexfacets) 
      average_normal_vector /= np.linalg.norm(average_normal_vector) 
      hashtable_nodal_normal[mapping[vertexindex]] = average_normal_vector 
   return hashtable_nodal_normal   

def nodal_contact_stiffness(node, surface_element, distance, gap):

   x = []; y = [] 
   for i in surface_element.entities(0): 
      j = Vertex(bmesh, i).point()
      x.append(j.x())
      y.append(j.y())       
      
   e1 = np.array([x[1] - x[0], y[1] - y[0]])
   D = np.inner(e1,e1)
   l = sqrt(D) 
   n = np.array([y[1] - y[0], x[0] - x[1]])
   n = n/sqrt(np.inner(n, n))   
   t = np.array(np.array([node.point().x(), node.point().y()]) - np.array([x[0], y[0]])) - np.inner(n, [node.point().x() - x[0], node.point().y() - y[0]])*n #  # 
   r = np.inner(t, e1)/l
   
   N = [[n[0], n[1], (1-r)*n[0], (1-r)*n[1], r*n[0], r*n[1]]]
   N = np.array(N)
   t1 = [x[1] - x[0], y[1] - y[0]]
   T1 = [[x[1] - x[0], y[1] - y[0], -(1-r)*(x[1] - x[0]), - (1-r)*(y[1] - y[0]), -r*(x[1] - x[0]), -r*(y[1] - y[0])]]
   T1 = np.array(T1)
   N1 = [[0, 0, -n[0], -n[1], n[0], n[1]]]            
   N1 = np.array(N1)
   
   NodalStiffness = np.zeros((6, 6))
   NodalStiffness = Penalty*np.heaviside(gap, 0.0)*N*N.transpose() - gap/l*(T1*N1.transpose() + N1*T1.transpose()) - gap*gap/l/l*N1*N1.transpose()     
   
   force = - Penalty*gap*l
   force1  =  Penalty*gap*(1 - r)*l
   force2  =  Penalty*gap*(r)*l
   Force = [force, force1, force2]
   
   return NodalStiffness, Force 
   

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
    
Vp = FunctionSpace(mesh, 'CG', p_element_degree)
Vu = VectorFunctionSpace(mesh, 'CG', u_element_degree)
bcpvz = DirichletBC(Vp, Constant(p_baseline_val), boundaries, 1)
bcpsz = DirichletBC(Vp, Constant(p_baseline_val), boundaries, 2)
bcusz = DirichletBC(Vu, u_vent, skull_boundary)
bcs_fluid = [bcpvz, bcpsz]
bcs_solid = [bcusz] 

p = TrialFunction(Vp)
u = Function(Vu)  
u_n = Function(Vu) 
p_n = Function(Vp)
vu = TestFunction(Vu)
vp = TestFunction(Vp)
g = u.geometric_dimension()


dofs0 = Vu.sub(0).dofmap().dofs(mesh, 0)
dofs1 = Vu.sub(1).dofmap().dofs(mesh, 0)

number_nodes = 0
for i in vertices(mesh): 
   number_nodes += 1      
print(number_nodes)
   
#Governing Equations
LHS = (k_water/viscosity_water)*dot(grad(p), grad(vp))*dx
RHS = 2*n_b*(L_p/R_cap)*(p_blood_val - p - tao*os_p)*vp*dx(1)
F = LHS - RHS

LHS = lhs(F)
RHS = rhs(F)

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
p = Function(Vp) 
lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, True #bicgstab
p = fe_mod.solve_lin_sys(Vp,LHS,RHS,bcs_fluid,lin_solver,precond,rtol,mon_conv,init_sol) 
pmax = p.vector().get_local().max()
print('pmax:', pmax)
b = plot(p)
plt.colorbar(b)
plot(mesh)
plt.show()

#f = Constant((0, -10))

u = TrialFunction(Vu)
RHS_solid = - dot(grad(p),vu)*dx
LHS_solid = inner(sigma(u + u_n), epsilon(vu))*dx #+ penaltycoeff*(vertex_distance_to_boundary_function)*dot(n,vu)*ds    
F_solid = LHS_solid - RHS_solid
RHS = rhs(F_solid) 
LHS = lhs(F_solid) 
u = Function(Vu)
   
lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, True #bicgstab
u = fe_mod.solve_lin_sys(Vu,LHS,RHS,bcs_solid,lin_solver,precond,rtol,mon_conv,init_sol) 
   
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, Vp)
u_magnitude_max=u_magnitude.vector().get_local().max()
      
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

distance_max = 1

RHS = rhs(F_solid) 
LHS = lhs(F_solid) 
K0 = assemble(LHS)
F0 = assemble(RHS)

K = PETScMatrix()
assemble(LHS, tensor=K)

#F = Vector
#F.set_local(np.array([1,2,3],dtype=np.float64))
F = as_backend_type(F0)
print(type(F))
#petF.createDense(F.shape,array=F0)
#petMat.setUp()
#F = PETScVector(petF) 

rtol = 1e-06
atol = 1e-09
  
res = np.allclose(K0.array(), K.array(), rtol, atol) 
print ("Are the two arrays are equal within the tolerance: \t", res) 
res = np.allclose(F0.get_local(), F.get_local(), rtol, atol) 
print ("Are the two arrays are equal within the tolerance: \t", res) 

while u_magnitude_max >= 5E-4 or distance_max > 5e-4: #or step<6:

   K = PETScMatrix()
   assemble(LHS, tensor=K)
   F = as_backend_type(F0)
   
   distance_record = []
   contact_forces = np.zeros(2*number_nodes)
   contact_stiffness_matrices = np.zeros((2*number_nodes, 2*number_nodes))
   
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
   print(len(contact_vertices))
   
   if len(contact_vertices) > 0: 
      hashtable_slave_master = dict()
      hashtable_slave_master_node = dict()
   #Find Closest Boundary to a Penetrating Vertex
      for v_idx in contact_vertices:
          v = Vertex(mesh, v_idx) 
          cell_idx = collisions_vertices[v_idx]
          contain_cell = Cell(mesh, cell_idx)
          cell_vertex_id = contain_cell.entities(0) 

          #Boundary Detect
          while not (list(set(cell_vertex_id) & set(contact_vertices))):  
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
       
          #Boundary Vertex to Create BoundaryMesh 
          sub_vertex = (list(set(cell_vertex_id) & set(contact_vertices))) 
          boundaries_sub = MeshFunction("size_t", bmesh, bmesh.topology().dim())
          boundaries_sub.set_all(0)    
          for v_sub_idx in sub_vertex:
              v_sub = boundary_backto_directoryb[boundary_backto_directory.index(v_sub_idx)]       
              v_sub = Vertex(bmesh, v_sub)
              for cell_sub in cells(v_sub): 
                  boundaries_sub[cell_sub] = 1000
                 
          submesh_sub = SubMesh(bmesh,boundaries_sub,1000) 
          vertex_indices = submesh_sub.data().array("parent_vertex_indices", 0)
          cell_indices = submesh_sub.data().array("parent_cell_indices", 1)

          bbtree_sub = BoundingBoxTree()
          bbtree_sub.build(submesh_sub)
          surface_element, distance = bbtree_sub.compute_closest_entity(v.point())
                    
          vertex_distance_to_boundary_function.vector()[V2D[v_idx]] += Penalty*distance
          distance_record.append(distance)
          
          hashtable_slave_master[v_idx] = cell_indices[surface_element]
          #print(cell_indices[surface_element])
          m_node = []
          m_node.append(dofs0[v_idx])
          m_node.append(dofs1[v_idx])
          for i in Cell(bmesh, cell_indices[surface_element]).entities(0):
              #print(i)
              m_node.append(dofs0[mapping[i]])
              m_node.append(dofs1[mapping[i]])
          #print(v_idx)    
          hashtable_slave_master_node[v_idx] = m_node
              
          aug_node = vertex_distance_to_boundary_function.vector()[V2D[v_idx]]  
          #print(aug_node, distance) 
          
          nodal_norm = nodal_normal(bmesh)   
              
          nodal_sf, nodal_force = nodal_contact_stiffness(Vertex(mesh, v_idx), Cell(bmesh, surface_element), aug_node, distance) 
                    
          contact_stiffness_matrix = np.zeros((2*number_nodes, 2*number_nodes))
          #print(nodal_sf, nodal_force)
          
          for i in range(6):
             for j in range(6):                 
                contact_stiffness_matrix[hashtable_slave_master_node[v_idx][i], hashtable_slave_master_node[v_idx][j]] += nodal_sf[i][j]
                          
          contact_force = np.zeros(2*number_nodes)
          #print('normal', nodal_norm[v_idx])
          
          for i in range(6):           
             contact_force[hashtable_slave_master_node[v_idx][i]] += nodal_force[i//2]*nodal_norm[v_idx][i%2]          
                    
          #Global Stiffness Matrix      
          contact_forces += contact_force 
          contact_stiffness_matrices += contact_stiffness_matrix 
          #print(contact_forces.shape, F.get_local().shape) 
          #print(contact_stiffness_matrices.shape, K.array().shape)
      print(type(F), type(contact_forces)) 
   
      for i in range(len(F)): 
         F[i] += contact_forces[i]
      print(type(F))
      print('start')
      petMat = PETSc.Mat()
      petMat.createDense(contact_stiffness_matrices.shape,array=contact_stiffness_matrices)
      petMat.setUp()
      mat = PETScMatrix(petMat) 
      print('end')
      print(mat.array().shape, K.array().shape)
      print(type(mat), type(K))    
      K += mat 
      #K.assemble()
      print(type(K))
      '''
      number_ = 0
      
      for i in vertices(bmesh): 
         
         if contact_forces[dofs0[mapping[i.index()]]] !=0 or contact_forces[dofs1[mapping[i.index()]]] !=0:
            print(F.get_local()[dofs0[mapping[i.index()]]], F.get_local()[dofs1[mapping[i.index()]]]) 
            print(contact_forces[dofs0[mapping[i.index()]]], contact_forces[dofs1[mapping[i.index()]]]) 
            #print(contact_stiffness_matrices[dofs0[mapping[i.index()]]][dofs1[mapping[i.index()]]], contact_stiffness_matrices[dofs1[mapping[i.index()]]][dofs0[mapping[i.index()]]])  
            #print('!!!', K.array()[dofs0[mapping[i.index()]]][dofs1[mapping[i.index()]]])   
            #K.array()[dofs0[mapping[i.index()]]][dofs1[mapping[i.index()]]] += contact_stiffness_matrices[dofs0[mapping[i.index()]]][dofs1[mapping[i.index()]]]
            #print(K.array()[dofs0[mapping[i.index()]]][dofs1[mapping[i.index()]]]) 
            print('!')
            number_ += 1
         else:
            print(mat.array()[dofs0[mapping[i.index()]]][dofs1[mapping[i.index()]]])
            print(K.array()[dofs0[mapping[i.index()]]][dofs1[mapping[i.index()]]])      
      print(number_)  
      '''        
      bcusz.apply(K)
      bcusz.apply(F)
      #solver = LUSolver(K)
      #solver.solve(K, u.vector(), F)
      solve(K, u.vector(), F, "gmres", "ilu")
      
      #u = project(u*0.1, Vu)   
      ALE.move(mesh, u)
      u_n.assign(u + u_n) #
   
      u_magnitude = sqrt(dot(u, u))
      u_magnitude = project(u_magnitude, Vp)
      b = plot(u_magnitude)
      plt.colorbar(b)
      plot(mesh)
      plt.show()
        
      # the Distance function
      distance_max = np.max(distance_record)
      u_magnitude_max = u_magnitude.vector().get_local().max()
      lambda_max = vertex_distance_to_boundary_function.vector().get_local().max()
      
      print('maximum displacement in the step = ', u_magnitude_max, 'maximum penetration depth = ',np.max(distance_record), 'LAMBDA', lambda_max)
      step += 1
      
   else: 
      print('maximum displacement in the step = ', u_magnitude_max, 'maximum penetration depth = ',np.max(distance_record), 'LAMBDA', lambda_max)
      step += 1
      
                
#Plot Results
c = plot(vertex_distance_to_boundary_function)
plt.colorbar(c)
plt.show()

u_magnitude = sqrt(dot(u_n, u_n))
u_magnitude = project(u_magnitude, Vp)
b = plot(u_magnitude)
plt.colorbar(b)
plot(mesh)
plt.show()

        
