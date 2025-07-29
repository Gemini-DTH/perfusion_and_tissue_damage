#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fenics
from fenics import *
import numpy as np 
import dolfin
import copy
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from scipy.sparse import csr_matrix, csr_array
import time
import itertools
import matplotlib.pyplot as plt
import sys

set_log_active(False)

#Parameters for the Poroelastic Model
p_element_degree = 1
u_element_degree = 1
p_baseline_val = 1330
nu = 0.35 # Poisson ratio
G = 216.3*2*1.37 
viscosity_blood = 3.6E-3 # Blood viscosity pa.s
k_water = 3.6E-15 # Water conductivity
viscosity_water = 1E-3 # Water viscosity
u_vent = Constant((0.0, 0.0, 0.0)) 
mu = Constant(2*G)   	
lmbda = Constant(2*G/(1.0 - 2.0*nu))

# Mesh
mesh = Mesh("mesh.xml")
md = MeshFunction("size_t", mesh, "mesh_func.xml") 
mf = MeshFunction("size_t", mesh, "mesh_funcf.xml")

#mesh1 = Mesh(mesh) 
mesh.init(0,1)
bmesh = BoundaryMesh(mesh, 'exterior')
bmesh.init(0,2)
mapping = bmesh.entity_map(0)

#Boundary Mesh and Mapping
adjacentNode = []
number_nodes = 0

t0 = time.time()

for v_idx in vertices(mesh):
   v = v_idx.index()
   number_nodes += 1
   neighborhood = []
   for cell_nei in v_idx.entities(1):                
      cell_neibh = Edge(mesh, cell_nei) 
      neigh = list((cell_neibh.entities(0)))
      neighborhood.extend(neigh) 
   adjacentNode.append(list(set(neighborhood))) 
         
print('mesh read done', number_nodes)

t1 = time.time()
print('1', t1-t0)

#bmesh = BoundaryMesh(mesh, "exterior")
#bmesh.init(0,2)
#mapping = bmesh.entity_map(0)
#bv_to_bf = dict()
hashtable_inverse_mapping = dict()

bbb = []
boundary_backto_directory = []
boundary_backto_directoryb = []

for i in vertices(bmesh): 
   bbb.append(mapping[i.index()])
   boundary_backto_directory.append(mapping[i])
   boundary_backto_directoryb.append(i.index()) 
   
with XDMFFile("bmesh.xdmf") as cfile:
   cfile.write(bmesh)   
'''   
# test diff
mesh.init(0,2)
bv_to_bf = dict()
   
t2 = time.time()
print('2', t2-t1)

#surfacefacets = []
surfacefacets_idx = []

t3 = time.time()
print('3', t3-t2)
     
mesh.init(0,2)   
for i in facets(mesh): 
   #print('a')
   if len(set(i.entities(0)) & (set(bbb))) == 3: 
      #print('aa')
      #surfacefacets.append(i)              
      surfacefacets_idx.append(i.index())
      #print('aaa')
      
boundary_vertex_id = []
for vertexindex in vertices(bmesh): 
   v0 = Vertex(mesh, mapping[vertexindex.index()])
   vertexfacets = []
   v0facets = set.intersection(set(v0.entities(2)), surfacefacets_idx)
   for i in v0facets:    
      vertexfacets.append(Facet(mesh, i))
   bv_to_bf[vertexindex.index()] = vertexfacets  
   hashtable_inverse_mapping[mapping[vertexindex.index()]] = vertexindex.index()
   boundary_vertex_id.append(mapping[vertexindex.index()])         
# test done
'''      
t2 = time.time()
print('2', t2-t1)

boundary_vertex_id = []
for vertexindex in vertices(bmesh): 
   #v0 = Vertex(mesh, mapping[vertexindex.index()])
   #vertexfacets = []
   #v0facets = set.intersection(set(v0.entities(2)), surfacefacets_idx)
   #for i in v0facets:    
   #   vertexfacets.append(Facet(mesh, i))
   #bv_to_bf[vertexindex.index()] = vertexfacets  
   hashtable_inverse_mapping[mapping[vertexindex.index()]] = vertexindex.index()
   boundary_vertex_id.append(mapping[vertexindex.index()])   
   
t3 = time.time()
print('3', t3-t2)

def nodal_normal(node):
   norms = np.zeros(3)
   for i in cells(node): 
      boundary_v = set.intersection(set(i.entities(0)), set(bbb))
      #print(len(boundary_v), len(i.entities(0)))
      if len(boundary_v) == 3: 
         inner_v = list(set(i.entities(0)) - boundary_v)[0]
         x_nn = []; y_nn = []; z_nn = []
         for j in boundary_v: 
            v = Vertex(mesh, j).point()
            x_nn.append(v.x())
            y_nn.append(v.y())
            z_nn.append(v.z())
         e1 = np.array((x_nn[1] - x_nn[0], y_nn[1] - y_nn[0], z_nn[1] - z_nn[0])) 
         e2 = np.array((x_nn[2] - x_nn[0], y_nn[2] - y_nn[0], z_nn[2] - z_nn[0]))  
         norm = np.cross(e1, e2)
         norm = norm/sqrt(np.inner(norm, norm))
         inner_vec = - np.array((Vertex(mesh, inner_v).point().x() - x_nn[0], Vertex(mesh, inner_v).point().y() - y_nn[0], Vertex(mesh, inner_v).point().z() - z_nn[0])) 
         if np.inner(inner_vec, norm) < 0: 
            norm = -norm
         #print(norms, norm)
         norms = norms + norm
         #print(norms)	
         
      if len(boundary_v) == 4: 
         surr_v = set(boundary_v - set(list([int(node.index())])))
         x_nn = []; y_nn = []; z_nn = []
         v = node.point()
         x_nn.append(v.x())
         y_nn.append(v.y())
         z_nn.append(v.z())                  
         for j in surr_v: 
            v = Vertex(mesh, j).point()
            x_nn.append(v.x())
            y_nn.append(v.y())
            z_nn.append(v.z())               
         e1 = np.array((x_nn[1] - x_nn[0], y_nn[1] - y_nn[0], z_nn[1] - z_nn[0])) 
         e2 = np.array((x_nn[2] - x_nn[0], y_nn[2] - y_nn[0], z_nn[2] - z_nn[0]))                
         e3 = np.array((x_nn[3] - x_nn[0], y_nn[3] - y_nn[0], z_nn[3] - z_nn[0]))    
         norm1 = np.cross(e1, e2)
         norm1 = norm1/sqrt(np.inner(norm1, norm1))
         inner_vec1 = - np.array((x_nn[3] - x_nn[0], y_nn[3] - y_nn[0], z_nn[3] - z_nn[0])) 
         if np.inner(inner_vec1, norm1) < 0: 
            norm1 = -norm1
         norms = norms + norm1 
            
         norm1 = np.cross(e3, e2)
         norm1 = norm1/sqrt(np.inner(norm1, norm1))
         inner_vec1 = - np.array((x_nn[1] - x_nn[0], y_nn[1] - y_nn[0], z_nn[1] - z_nn[0])) 
         if np.inner(inner_vec1, norm1) < 0: 
            norm1 = -norm1
         norms = norms + norm1 
            
         norm1 = np.cross(e1, e3)
         norm1 = norm1/sqrt(np.inner(norm1, norm1))
         inner_vec1 = - np.array((x_nn[2] - x_nn[0], y_nn[2] - y_nn[0], z_nn[2] - z_nn[0])) 
         if np.inner(inner_vec1, norm1) < 0: 
            norm1 = -norm1                            
         norms = norms + norm1 
           
   norms = norms/np.linalg.norm(norms)           
   return norms      

def nodal_contact_stiffness(node, surface_element, aug_node, gap, r0, s0):
   m_node = []
   m_node.append(dofs0[node.index()])
   m_node.append(dofs1[node.index()])
   m_node.append(dofs2[node.index()])
   x = []; y = []; z = []
   for i in surface_element.entities(0): 
      j = Vertex(bmesh, i).point()
      x.append(j.x())
      y.append(j.y())
      z.append(j.z())
      m_node.append(dofs0[mapping[i]])     
      m_node.append(dofs1[mapping[i]])
      m_node.append(dofs2[mapping[i]])  
   e1 = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
   e2 = np.array([x[2] - x[0], y[2] - y[0], z[2] - z[0]])
   e3 = np.array([x[2] - x[1], y[2] - y[1], z[2] - z[1]])
   D = np.inner(e1,e1)*np.inner(e2, e2) - np.inner(e1, e2)*np.inner(e2, e1)
   n = np.cross(e1, e2)
   n = n/sqrt(np.inner(n, n))   
   t = np.array(np.array([node.point().x(), node.point().y(), node.point().z()]) - np.array([x[0], y[0], z[0]])) - np.inner(n, [node.point().x() - x[0], node.point().y() - y[0], node.point().z() - z[0]])*n
   r = np.inner(t, e1)/D*np.inner(e1, e1) - np.inner(t, e2)/D*np.inner(e2, e1) 
   s = - np.inner(t, e1)/D*np.inner(e1, e2) + np.inner(t, e2)/D*np.inner(e2, e2) 
   r = r0; s = s0
       
   N = np.array([[n[0], n[1], n[2], (1-r-s)*n[0], (1-r-s)*n[1], (1-r-s)*n[2], r*n[0], r*n[1], r*n[2], s*n[0], s*n[1], s*n[2]]])
   t1 = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
   t2 = [x[2] - x[0], y[2] - y[0], z[2] - z[0]]
   T1 = np.array([[x[1] - x[0], y[1] - y[0], z[1] - z[0], 
         - (1 -r -s)*(x[1] - x[0]), - (1 -r -s)*(y[1] - y[0]), - (1 -r -s)*(z[1] - z[0]),
         -r*(x[1] - x[0]), -r*(y[1] - y[0]), -r*(z[1] - z[0]),
         -s*(x[1] - x[0]), -s*(y[1] - y[0]), -s*(z[1] - z[0])]])
   T2 = np.array([[x[2] - x[0], y[2] - y[0], z[2] - z[0], 
         - (1 -r -s)*(x[2] - x[0]), - (1 -r -s)*(y[2] - y[0]), - (1 -r -s)*(z[2] - z[0]),
         -r*(x[2] - x[0]), -r*(y[2] - y[0]), -r*(z[2] - z[0]),
         -s*(x[2] - x[0]), -s*(y[2] - y[0]), -s*(z[2] - z[0])]]) 
     
   N1 = np.array([[0, 0, 0, -n[0], -n[1], -n[2], n[0], n[1], n[2], 0, 0, 0]])
   N2 = np.array([[0, 0, 0, -n[0], -n[1], -n[2], 0, 0, 0, n[0], n[1], n[2]]])           
   
   M = np.zeros((2,2))
   M[0][0] = np.inner(t1, t1); M[0][1] = np.inner(t1, t2)
   M[1][0] = np.inner(t1, t2); M[1][1] = np.inner(t2, t2)
   
   detA = M[0][0]*M[1][1] - M[0][1]*M[1][0]
   D1 = (1/detA)*(M[1][1]*(T1 + gap*N1) - M[0][1]*(T2 + gap*N2));
   D2 = (1/detA)*(M[0][0]*(T1 + gap*N2) - M[0][1]*(T1 + gap*N1));      
       
   l1 = sqrt(np.inner(e1, e1)); l2 = sqrt(np.inner(e2, e2)); l3 = sqrt(np.inner(e3, e3))
   semi_peri = (l1+l2+l3)/2 
   a = (sqrt(semi_peri*(semi_peri-l1)*(semi_peri-l2)*(semi_peri-l3)))
   
   NodalStiffness = np.zeros((12, 12))
   NodalStiffness = Penalty*a*np.heaviside((aug_node + Penalty*gap), 0)*N*N.transpose() - np.heaviside((aug_node + Penalty*gap), 0)*(aug_node*a + Penalty*gap*a)*(D1*N1.transpose() + D2*N2.transpose() + N1*D1.transpose() + N2*D2.transpose()) + np.heaviside((aug_node + Penalty*gap), 0)*(aug_node*a + Penalty*gap*a)*gap*(M[0][0]*N1*N1.transpose() + M[0][1]*N1*N2.transpose() + M[1][0]*N2*N1.transpose() + M[1][1]*N2*N2.transpose())     
   force = - np.heaviside((Penalty*gap*a + aug_node*a), 0)*(Penalty*gap*a + aug_node*a)
   force1  =  np.heaviside((Penalty*gap*a + aug_node*a), 0)*(Penalty*gap*(1 - r - s)*a + aug_node*(1 - r - s)*a)
   force2  =  np.heaviside((Penalty*gap*a + aug_node*a), 0)*(Penalty*gap*r*a + aug_node*r*a)
   force3 = np.heaviside((Penalty*gap*a + aug_node*a), 0)*(Penalty*gap*s*a + aug_node*s*a)
   Force = [force, force1, force2, force3]
            
   return NodalStiffness, Force, m_node   

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

Penalty = 10000 # 300000 big # 3000 small # 80000 big # 30000 big

def distance_measure(node, cells):
   distance_tmp = 1
   distance_tmp0 = 1
   dislist = []; rlist = []; slist = []; olist = []; ele = []
   master_surface_ele = 0.1
   master_surface_ele0 = 0.1 
   rn = 0; sn = 0; on = 0
   rx = 0; sx = 0; ox = 0
   for i in cells: 
      i_cell = Cell(bmesh, i.index())
      x = []; y = []; z = []
      for k in i_cell.entities(0): 
         j = Vertex(bmesh, k).point()
         x.append(j.x())
         y.append(j.y())
         z.append(j.z())  
      e1 = np.array([x[1] - x[0], y[1] - y[0], z[1] - z[0]])
      e2 = np.array([x[2] - x[0], y[2] - y[0], z[2] - z[0]])
      n = np.cross(e1, e2)
      n = n/sqrt(np.inner(n, n))  
      distance0 = np.absolute(np.inner(n, [node.point().x() - x[0], node.point().y() - y[0], node.point().z() - z[0]]))   
      r0 = np.array([node.point().x() - x[0], node.point().y() - y[0], node.point().z() - z[0]])
      r1 = np.array([node.point().x() - x[1], node.point().y() - y[1], node.point().z() - z[1]])
      r2 = np.array([node.point().x() - x[2], node.point().y() - y[2], node.point().z() - z[2]])
      D = np.inner(e1,e1)*np.inner(e2, e2) - np.inner(e1, e2)*np.inner(e2, e1)
      n = np.cross(e1, e2)
      n = n/sqrt(np.inner(n, n))   
      t = np.array(np.array([node.point().x(), node.point().y(), node.point().z()]) - np.array([x[0], y[0], z[0]])) - np.inner(n, [node.point().x() - x[0], node.point().y() - y[0], node.point().z() - z[0]])*n
      
      r = np.inner(t, e1)/D*np.inner(e1, e1) - np.inner(t, e2)/D*np.inner(e2, e1) 
      s = - np.inner(t, e1)/D*np.inner(e1, e2) + np.inner(t, e2)/D*np.inner(e2, e2)    
          
      if r > 0 and s > 0 and (1 - r - s) > 0: 
         dislist.append(distance0); rlist.append(r); slist.append(s); olist.append(1-r-s); ele.append(i.index())
   
   if len(dislist) == 0: 
      for i in cells: 
         i_cell = Cell(bmesh, i.index())   
         x_edge = []; y_edge = []; z_edge = []        
         for n in i_cell.entities(0):
            edge_p = Vertex(bmesh, n)
            x_edge.append(edge_p.point().x())
            y_edge.append(edge_p.point().y())
            z_edge.append(edge_p.point().z())
            
         ts = [np.array([node.point().x() - x_edge[0], node.point().y() - y_edge[0], node.point().z() - z_edge[0]]), np.array([node.point().x() - x_edge[1], node.point().y() - y_edge[1], node.point().z() - z_edge[1]]), np.array([node.point().x() - x_edge[2], node.point().y() - y_edge[2], node.point().z() - z_edge[2]])]

         es = [np.array([x_edge[1] - x_edge[0], y_edge[1] - y_edge[0], z_edge[1] - z_edge[0]]), np.array([x_edge[2] - x_edge[1], y_edge[2] - y_edge[1], z_edge[2] - z_edge[1]]), np.array([x_edge[0] - x_edge[2], y_edge[0] - y_edge[2], z_edge[0] - z_edge[2]])]
         e_norm = [np.linalg.norm(es[0]), np.linalg.norm(es[1]), np.linalg.norm(es[2])]
                           
         os = [1 - np.inner(ts[0], es[0])/e_norm[0], 0, np.inner(ts[2], es[2])/e_norm[2]]         
         rs = [np.inner(ts[0], es[0])/e_norm[0],1 - np.inner(ts[1], es[1])/e_norm[1], 0]
         ss = [0, np.inner(ts[1], es[1])/e_norm[1], 1 - np.inner(ts[2], es[2])/e_norm[2]]

         distances = [np.linalg.norm(ts[0] - rs[0]*es[0]/e_norm[0]), np.linalg.norm(ts[1] - ss[1]*es[1]/e_norm[1]), np.linalg.norm(ts[2] - os[2]*es[2]/e_norm[2])]         
          
         for m in range(3):  
            if rs[m]>0 and rs[m]<1: 
               dislist.append(distances[m]); rlist.append(rs[m]); slist.append(ss[m]); olist.append(os[m]); ele.append(i.index())   
         
   if len(dislist) == 0:      
      for i in cells: 
         i_cell = Cell(bmesh, i.index())   
         x_node = []; y_node = []; z_node = []
         for k in i_cell.entities(0): 
            j = Vertex(bmesh, k).point()
            x_node.append(j.x()); 
            y_node.append(j.y())
            z_node.append(j.z())
         
         distances = [np.linalg.norm([node.point().x() - x_node[0], node.point().y() - y_node[0], node.point().z() - z_node[0]]), np.linalg.norm([node.point().x() - x_node[1], node.point().y() - y_node[1], node.point().z() - z_node[1]]), np.linalg.norm([node.point().x() - x_node[2], node.point().y() - y_node[2], node.point().z() - z_node[2]])]
         distance0 = min([np.linalg.norm([node.point().x() - x_node[0], node.point().y() - y_node[0], node.point().z() - z_node[0]]), np.linalg.norm([node.point().x() - x_node[1], node.point().y() - y_node[1], node.point().z() - z_node[1]]), np.linalg.norm([node.point().x() - x_node[2], node.point().y() - y_node[2], node.point().z() - z_node[2]])])
         
         minpos0 = distances.index(min(distances))
         dislist.append(distance0);    
         rlist.append(int(bool(minpos0 == 1))); slist.append(int(bool(minpos0 == 2))); olist.append(int(bool(minpos0 == 0))); ele.append(i.index())
         ele.append(i.index())
                  
   if len(dislist) != 0:
      distance_tmp = min(dislist) 
      minpos = dislist.index(min(dislist))            
      master_surface_ele = ele[minpos] #i.index()
      rn = rlist[minpos]; sn = slist[minpos]; on = olist[minpos]                              
   
   else:
      rn = 0; sn = 0; on = 0; master_surface_ele = i.index()
      distance_tmp = 0
      print('rs warning!!!')  
   
   if distance_tmp > 0.02: 
      print('distance_error', node.point().x(), node.point().y(),node.point().z())   
      for i in master_surface_ele.entities(0): 
         j = Vertex(bmesh, i)
         print(j.point().x(), j.point().y(), j.point().z())  
               
   return master_surface_ele, distance_tmp, rn, sn                   


#bbtree = mesh.bounding_box_tree()
#bbtree.build(mesh)   
#bbtreecom = mesh1.bounding_box_tree() 
#bbtreecom.build(mesh1)

#bbtree = BoundingBoxTree()
#bbtree.build(bmesh)

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
    
# Define Distance Function
Vk_function = FunctionSpace(mesh, 'CG', 1)
vertex_distance_to_boundary_function = Function(Vk_function)
distance_test = Function(Vk_function)
V2D = vertex_to_dof_map(Vk_function) 

# Compute solution and Computational parameters to initialize
#Simulation and Move Mesh   
Vp = FunctionSpace(mesh, 'CG', p_element_degree)
p = Function(Vp) 
fFile = HDF5File(MPI.comm_world,"pn3886.h5","r") #n3961
fFile.read(p,"/f")
fFile.close()

Vu = VectorFunctionSpace(mesh, 'CG', u_element_degree)
u = Function(Vu)  
u_n = Function(Vu) 
vu = TestFunction(Vu)
g = u.geometric_dimension()
dofs0 = Vu.sub(0).dofmap().dofs(mesh, 0)
dofs1 = Vu.sub(1).dofmap().dofs(mesh, 0)
dofs2 = Vu.sub(2).dofmap().dofs(mesh, 0)

#Governing Equations and Move Mesh 
u_n = Function(Vu)  
u = TrialFunction(Vu)
RHS_solid = - dot(grad(p),vu)*dx
LHS_solid = inner(sigma(u + u_n), epsilon(vu))*dx 
F_solid = LHS_solid - RHS_solid
RHS = rhs(F_solid)  
LHS = lhs(F_solid) 

u = Function(Vu)  

F0 = assemble(RHS)
K = PETScMatrix()
assemble(LHS, tensor=K)
F = as_backend_type(F0)

for i in [1,2,3,4,5,6,8]: #[2]:#
   bc = DirichletBC(Vu, u_vent, mf, i)
   bc.apply(K)
   bc.apply(F)
   
solve(K, u.vector(), F, "bicgstab", "amg") #"gmres", "ilu")

t5 = time.time()
print('Solve', t5-t3)

u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, Vp, solver_type='bicgstab', preconditioner_type='amg')
u_magnitude_max = u_magnitude.vector().get_local().max() 

# Relaxation
u = project(u*0.7, Vu, solver_type='bicgstab', preconditioner_type='amg')   
   
ALE.move(mesh, u)
u_n.assign(u + u_n) 
file = File('u_test.pvd')
file << u_magnitude 
print('maximum displacement in the step = ', u_magnitude_max) 

bmesh = BoundaryMesh(mesh, 'exterior')
bmesh.init(0,2)

#mapping0 = bmesh.entity_map(0)
#for i in vertices(bmesh): 
#   assert mapping0[i.index()] == mapping[i.index()]

hashtable_n2f = dict()
hashtable_n2n = dict()
hashtable_n2norm = dict()

step = 0
distance_max = 1

mesh.init(0,3)
 
step = 0
while distance_max > 5e-4:  
   sub_step = 0;  u_magnitude_max = 5.1e-4 
   while u_magnitude_max >= 5e-4: 
      t0 = time.time()
      
      u = TrialFunction(Vu)
      RHS_solid = - dot(grad(p),vu)*dx
      LHS_solid = inner(sigma(u + u_n), epsilon(vu))*dx 
      RHS = rhs(F_solid) 
      LHS = lhs(F_solid) 
      
      K = assemble(LHS)
      F0 = assemble(RHS)

      for i in [1,2,3,4,5,6,8]: #[2]:# 
         bc = DirichletBC(Vu, u_vent, mf, i)    
         bc.apply(K)
         
      K = as_backend_type(K).mat()
      #bcusz.apply(F) 

      contact_forces = np.zeros(3*number_nodes)  
      contact_stiffness_matrices = csr_matrix((3*number_nodes, 3*number_nodes), dtype=np.float64)
      distance_record = dict()

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
   
      #Contact Boundary Vertex
      boundary_verticenum = boundary_backto_directory 
      contact_vertices = list(set(boundary_verticenum) & set(contact_point)) 
      print('incontact', len(contact_vertices))

      for v_idx in contact_vertices:
         v = Vertex(mesh, v_idx) 
         cell_idx = collisions_vertices[v_idx]
         contain_cell = Cell(mesh, cell_idx)
         cell_vertex_id = contain_cell.entities(0) 
     
         #Boundary Detect
         while not (list(set(cell_vertex_id) & set(boundary_verticenum))):  
            cell_vertex_id0 = copy.deepcopy(cell_vertex_id)
            neighborhood = []
            for cell_v_id in cell_vertex_id:
                neighborhood.extend(adjacentNode[cell_v_id])  
            cell_vertex_id = list(set(neighborhood))
            cell_vertex_id =  list(set(cell_vertex_id) - set(cell_vertex_id0))
            
         cell_vertex_id = (list(set(cell_vertex_id) & set(boundary_verticenum)))  

         while len(cell_vertex_id) < 5: 
            neighborhood = []
            for cell_v_id in cell_vertex_id:
                neighborhood.extend(adjacentNode[cell_v_id])  
            cell_vertex_id = list(set(neighborhood))
            cell_vertex_id = list(set(cell_vertex_id) & set(boundary_verticenum))                      

         #Boundary Vertex to Create BoundaryMesh          
         sub_vertex = (list(set(cell_vertex_id) & set(boundary_verticenum))) 
         boundaries_sub = MeshFunction("size_t", bmesh, bmesh.topology().dim())
         master_surface = []
         master_node_surface = [] 
         distance_ms_node = 1
         
         dis = 10
         for v_sub_idx in sub_vertex:
            v_sub = hashtable_inverse_mapping[v_sub_idx]
            v_sub = Vertex(bmesh, v_sub)
            dis_find = 0; count = 0            
            for i in cells(v_sub):
               for j in vertices(i):
                  count += 1
                  dis_find += np.linalg.norm([v.point().x() - j.point().x(), v.point().y() - j.point().y(), v.point().z() - j.point().z()])
            if dis_find/count < dis: 
               dis = dis_find/count
               master_vertex = v_sub                 
            for cell_sub in cells(v_sub): 
               master_surface.append(cell_sub)     
         for cell_sub in cells(master_vertex):
            master_node_surface.append(cell_sub) 
         node_cells = []   
         for i in cells(Vertex(bmesh, hashtable_inverse_mapping[v_idx])): 
            node_cells.append(i)   
                           

         hashtable_n2n[v_idx] = master_vertex         
         master_node_surface = list(set(master_node_surface) - set(node_cells))
         hashtable_n2f[v_idx] = master_node_surface          
         
         surface_element, distance, r_value, s_value = distance_measure(Vertex(mesh, v_idx), master_node_surface) 
         
         distance_record[v_idx] = distance      
 
         lag_aug = vertex_distance_to_boundary_function.vector()[V2D[v_idx]]  
         
         # test diff
         '''          
         vertexfacets = bv_to_bf[hashtable_inverse_mapping[v_idx]]
         
         nodal_norm = np.array([[facet.normal().x(),
                                      facet.normal().y(),
                                      facet.normal().z()]
                                      for facet in vertexfacets]
                                      ).mean(axis=0)
         nodal_norm /= np.linalg.norm(nodal_norm) 
         '''
         nodal_norm = nodal_normal(Vertex(mesh, v_idx))
         #test done
         
         #test norm
         hashtable_n2norm[v_idx] = nodal_norm 
              
         nodal_sf, nodal_force, ms_node = nodal_contact_stiffness(Vertex(mesh, v_idx), Cell(bmesh, surface_element), lag_aug, distance, r_value, s_value) 
                       
         nodal_sf = nodal_sf.flatten()                  
         I = np.array(np.meshgrid(np.array(ms_node), np.array(ms_node))).T.reshape(-1, 2).flatten()       
         I1 = I[0::2]; I2 = I[1::2]
         contact_force = np.outer(np.array(nodal_force), np.array(nodal_norm)).reshape(-1)  
         mtx = csr_matrix((nodal_sf, (I1, I2)), shape=(3*number_nodes, 3*number_nodes)) 
         contact_stiffness_matrices += mtx
                                 
         np.add.at(contact_forces, np.array(ms_node), contact_force)           
      
      not_in_contact = list(set(hashtable_n2f.keys())-set(contact_vertices))    
      print('notin', len(not_in_contact))  
      for v_idx in not_in_contact:
         sub_cells = hashtable_n2f[v_idx]
         surafce_element, distance, r_value, s_value = distance_measure(Vertex(mesh, v_idx), sub_cells) 
         distance_record[v_idx] = - distance 
         
         lag_aug = vertex_distance_to_boundary_function.vector()[V2D[v_idx]]            
         '''
         # test diff
         vertexfacets = bv_to_bf[hashtable_inverse_mapping[v_idx]]
          
         nodal_norm = np.array([[facet.normal().x(),
                                      facet.normal().y(),
                                      facet.normal().z()]
                                      for facet in vertexfacets]
                                      ).mean(axis=0)
         nodal_norm /= np.linalg.norm(nodal_norm)
         '''
         nodal_norm = nodal_normal(Vertex(mesh, v_idx))
         #test done 
                    
         nodal_sf, nodal_force, ms_node = nodal_contact_stiffness(Vertex(mesh, v_idx), Cell(bmesh, surface_element), lag_aug, - distance, r_value, s_value) 
                            
         contact_force = np.zeros(3)            
         contact_force = np.array([nodal_force[0]*nodal_norm[0], nodal_force[0]*nodal_norm[1], nodal_force[0]*nodal_norm[2]])    
         np.add.at(contact_forces, np.array([ms_node[0], ms_node[1], ms_node[2]]), contact_force)     

      F0.add_local(contact_forces)
      
      for i in [1,2,3,4,5,6,8]: #[2]:#
         bc = DirichletBC(Vu, u_vent, mf, i)    
         bc.apply(F0)
                      
      t1 = time.time()
      print('coord calc time', t1-t0)
      '''
      #rol, col, val = K.data() 
      #del K      
      '''   
      contact_stiffness_matrices = csr_matrix(K.getValuesCSR()[::-1], shape=K.size) + contact_stiffness_matrices 
      t2 = time.time()        
      print('add array', t2-t1) 
      
      del K #del rol; del col; del val                    

      csr = (contact_stiffness_matrices.indptr, contact_stiffness_matrices.indices, contact_stiffness_matrices.data)
      petMat = PETSc.Mat().createAIJ(size=contact_stiffness_matrices.shape, csr=csr)
      petMat.assemble()
      
      del contact_stiffness_matrices

      mat = PETScMatrix(petMat)
     
      del petMat       
                
      mat = Matrix(mat)
      print('NNZ', mat.nnz())      

      t3 = time.time()
      print('assemble', t3-t2)                    
      print('solving...')
      u = Function(Vu)         
      solve(mat, u.vector(), F0, "gmres", "ilu") #"gmres", "ilu") # bicgstab
      
      print('max_contact_force', np.max(contact_forces))            
      
      t4 = time.time()
      print('solve time', t4-t3)
           
      u_magnitude = sqrt(dot(u, u))
      u_magnitude = project(u_magnitude, Vp, solver_type='bicgstab', preconditioner_type='amg')
      u_magnitude_max=u_magnitude.vector().get_local().max()
      print('max_u_step',u_magnitude_max)
      
      # Relaxation      
      u = project(u*0.5, Vu, solver_type='bicgstab', preconditioner_type='amg')  
      ALE.move(mesh, u)
      u_n.assign(u + u_n) 
      
      del mat
      
      bmesh = BoundaryMesh(mesh, 'exterior')
      bmesh.init(0,2)
      
      sub_step += 1
           
      file = File('utestBC' + str(sub_step)+'.pvd')
      file << u_magnitude    
           
      #u_magnitude_max = 0
      print(sub_step, 'substep')
      
      for v_idx in list(distance_record.keys()): 
          distance_test.vector()[V2D[v_idx]] = Penalty*distance_record[v_idx]
      file = File('distance_test' + str(sub_step)+'.pvd')
      file << distance_test

   distance_max = 0
   print('not zero', len(list(hashtable_n2f.keys())))
   if len(list(hashtable_n2f.keys())) != 0:
      for v_idx in list(distance_record.keys()):                
         vertex_distance_to_boundary_function.vector()[V2D[v_idx]] += Penalty*distance_record[v_idx]  # 
         if distance_record[v_idx] > distance_max: 
            distance_max = distance_record[v_idx]
 
      print(step, 'max_distance_step, ooooooooooooo', distance_max)
      
      u_magnitude = sqrt(dot(u_n, u_n))
      u_magnitude = project(u_magnitude, Vp, solver_type='bicgstab', preconditioner_type='amg')
      file = File('dis.pvd')
      file << vertex_distance_to_boundary_function 
      file = File('u' + str(step)+'.pvd')
      file << u_magnitude       
   
   else :
       
      distance_max = 0 
      print(step, 'max_distance_step, oooooooooooooo', distance_max)

   step += 1       
   #distance_max = 0 #
  
