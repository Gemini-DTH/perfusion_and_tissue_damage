#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
import meshio  #version 3.0.6

# Create mesh and define function space vol_mesh_1e6
msh = meshio.read("labelled_test_mesh_novent.msh") # fullbrainmesh.msh

#Write Mesh
meshio.write("mesh.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]}))
                               
meshio.write("mf.xdmf", meshio.Mesh(points=msh.points, cells={"triangle": msh.cells["triangle"]},
                                    cell_data={"triangle": {"name_to_read": msh.cell_data["triangle"]["gmsh:physical"]}}))

meshio.write("md.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]},
                                      cell_data={"tetra": {"name_to_read": msh.cell_data["tetra"]["gmsh:physical"]}}))


