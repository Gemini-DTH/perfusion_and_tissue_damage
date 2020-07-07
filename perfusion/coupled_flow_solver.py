"""
Multi-compartment Darcy flow model with mixed Dirichlet and Neumann
boundary conditions

System of equations (no summation notation)
Div ( Ki Grad(pi) ) - Sum_j=1^3 beta_ij (pi-pj) = sigma_i

Ki - permeability tensor [mm^3 s / g]
pi & pj - volume averaged pressure in the ith & jth comparments [Pa]
beta_ij - coupling coefficient between the ith & jth compartments [Pa / s]
sigma_i - source term in the ith compartment [1 / s]

@author: Tamas Istvan Jozsa
"""

#%% IMPORT MODULES
# installed python3 modules
from dolfin import *
import time
import sys
import argparse
import numpy
numpy.set_printoptions(linewidth=200)


# ghost mode options: 'none', 'shared_facet', 'shared_vertex'
parameters['ghost_mode'] = 'none'

# added module
import IO_fcts
import suppl_fcts
import finite_element_fcts as fe_mod

# location of the 1-D blood flow model
sys.path.insert(0, "../../1d-blood-flow/")
from Blood_Flow_1D import Patient, Results
import contextlib
import copy
import scipy.optimize

# solver runs is "silent" mode
set_log_level(50)

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

start0 = time.time()

#%% READ INPUT
if rank == 0: print('Step 1: Reading input files, initialising functions and parameters')
start1 = time.time()

parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--config_file", help="path to configuration file (string ended with /)",
                    type=str, default='./config_coupled_flow_solver.xml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                type=str, default=None)
config_file = parser.parse_args().config_file

configs = IO_fcts.basic_flow_config_reader2(config_file,parser)
# physical parameters
p_arterial, p_venous = configs.physical.p_arterial, configs.physical.p_venous
K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
    configs.physical.K1gm_ref, configs.physical.K2gm_ref, configs.physical.K3gm_ref, configs.physical.gmowm_perm_rat
beta12gm, beta23gm, gmowm_beta_rat = \
    configs.physical.beta12gm, configs.physical.beta23gm, configs.physical.gmowm_beta_rat

# read mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs.input.mesh_file)

# determine fct spaces
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, configs.simulation.fe_degr)

# initialise permeability tensors
K1, K2, K3 = IO_fcts.initialise_permeabilities(K1_space,K2_space,mesh,configs.input.permeability_folder)


if rank == 0: print('\t Scaling coupling coefficients and permeability tensors')

# set coupling coefficients
beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains, \
                                beta12gm, beta23gm, gmowm_beta_rat, \
                                K2_space, configs.output.res_fldr, configs.output.save_pvd)

K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1, K2, K3, \
                                  K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, \
                                  configs.output.res_fldr,configs.output.save_pvd)
end1 = time.time()


#%% SET UP FINITE ELEMENT SOLVER AND SOLVE GOVERNING EQUATIONS
if rank == 0: print('Step 2: Defining and solving governing equations')
start2 = time.time()

# 1-D blood flow model
# get patient folder location
patient_folder = "/".join(configs.input.inlet_boundary_file.split("/")[:-2]) + "/"  # assume boundary file is in bf_sim folder

# run 1-D blood flow model and update boundary file
coarseCollaterals=True
solver = "krylov"
clotactive = False

Patient = Patient.Patient(patient_folder)
Patient.LoadBFSimFiles()
Patient.LoadModelParameters("Model_parameters.txt")
Patient.LoadClusteringMapping(Patient.Folders.ModellingFolder + "Clusters.csv")

Patient.Initiate1DSteadyStateModel()  # run with original wk elements
Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True, coarseCollaterals=coarseCollaterals)
# boundary condition before coupling
# Patient.Perfusion.UpdateMappedRegionsFlowdata(inlet_boundary_file)

# save old flowrates
for index, node in enumerate(Patient.Topology.OutletNodes):
    # node.OldFlow = node.FlowRate
    node.OldFlow = node.WKNode.AccumulatedFlowRate

# set venous pressure
# for node in Patient.Topology.OutletNodes:
#     node.OutPressure = 0
UniformPressure = True
if UniformPressure:
    # uniform pressure
    Patient.UpdatePressureCouplingPoints(p_arterial)
else:
    # pressure drop fraction
    pressuredrop = 0.3  # fraction
    surfacepressure = [(1-pressuredrop)*cp.Node.Pressure for _, cp in enumerate(Patient.Perfusion.CouplingPoints)]
    Patient.UpdatePressureCouplingPoints(surfacepressure)

# update boundary file
for outlet in Patient.Topology.OutletNodes:
    outlet.Pressure = outlet.OutPressure
Patient.Perfusion.UpdateMappedRegionsFlowdata(configs.input.inlet_boundary_file)

# set up finite element solver
# TODO: handle Neuman/dirichlet boundary conditions
LHS, RHS, sigma1, sigma2, sigma3, BCs = \
fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, \
                         p, p1, p2, p3, K1, K2, K3, beta12, beta23, \
                         p_arterial, p_venous, \
                         configs.input.read_inlet_boundary, configs.input.inlet_boundary_file, configs.input.inlet_BC_type)

lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

# tested iterative solvers for first order elements: gmres, cg, bicgstab
#linear_solver_methods()
#krylov_solver_preconditioners()
if rank == 0: print('\t pressure computation')
p = fe_mod.solve_lin_sys(Vp,LHS,RHS,BCs,lin_solver,precond,rtol,mon_conv,init_sol)
end2 = time.time()


#%% COMPUTE VELOCITY FIELDS, SAVE SOLUTION, EXTRACT FIELD VARIABLES
if rank == 0: print('Step 3: Computing velocity fields, saving results, and extracting some field variables')
start3 = time.time()

p1, p2, p3=p.split()

perfusion = project(beta12 * (p1-p2)*6000,K2_space, solver_type='bicgstab', preconditioner_type='amg')

# compute velocities
vel1 = project(-K1*grad(p1),Vvel, solver_type='bicgstab', preconditioner_type='amg')
vel2 = project(-K2*grad(p2),Vvel, solver_type='bicgstab', preconditioner_type='amg')
vel3 = project(-K3*grad(p3),Vvel, solver_type='bicgstab', preconditioner_type='amg')

ps = [p1, p2, p3]
vels = [vel1, vel2, vel3]
Ks = [K1, K2, K3]

# get surface values
fluxes, surf_p_values = suppl_fcts.surface_ave(mesh, boundaries, vels, ps)

FlowRateAtBoundary = fluxes[:, 2][2:] * -1  # Flow rate from the perfusion model (sign to match 1-d bf model, positive flow towards the brain)
PressureAtBoundary = surf_p_values[:, 2][2:]  # Pressure from the perfusion model

# %% OPTIMIZE 1-D BLOOD FLOW MODEL
print("\tOptimize 1-D blood flow model to match perfusion model.")
# update boundary condition
for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
    cp.Node.TargetFlow = FlowRateAtBoundary[index] * 1e-3
# erorratio = [p_arterial / p for p in PressureAtBoundary]
# print(erorratio)

# optimize wk elements
# calibration step such that the 1d-bd and perfusion models agree on flowrate and pressure in the healthy scenario
rel_tol = 1e-5
relative_residual = 1
while relative_residual > rel_tol:
    oldR = numpy.array([node.Node.R1+node.Node.R2 for node in Patient.Perfusion.CouplingPoints])
    for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
        cp.Node.R1 = (cp.Node.Pressure - cp.Node.OutPressure) / (cp.Node.TargetFlow * 1e-6)
        cp.Node.R2 = 1
    with contextlib.redirect_stdout(None):
        Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True, coarseCollaterals=coarseCollaterals)
    # residual = [Node.Node.TargetFlow - Node.Node.WKNode.AccumulatedFlowRate for index, Node in
    #             enumerate(Patient.Perfusion.CouplingPoints)]
    # newr = numpy.array([node.Node.R1 for node in Patient.Perfusion.CouplingPoints])
    # diff = oldR - newr
    # sq_diff = sum(numpy.power(diff, 2))
    # print(f'\tResidual: {sq_diff}')
    relative_residual = max([abs(node.Node.R1+node.Node.R2-oldR[index])/(node.Node.R1+node.Node.R2) for index, node in enumerate(Patient.Perfusion.CouplingPoints)])
    print(f'\tMax relative residual: {relative_residual}')

# save optimization results and model parameters
with open(patient_folder + 'Model_values_Healthy.csv', "w") as f:
    f.write(
        "Region,Resistance,Outlet Pressure(pa),WK Pressure, Perfusion Surface Pressure(pa),Old Flow Rate,Flow Rate(mm^3/s),Perfusion Flow Rate(mm^3/s)\n")
    for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
        f.write("%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n" % (
            fluxes[:, 0][2:][index],
            cp.Node.R1,
            cp.Node.Pressure,
            cp.Node.WKNode.Pressure,
            PressureAtBoundary[index],
            cp.Node.OldFlow,
            # cp.Node.FlowRate,
            cp.Node.WKNode.AccumulatedFlowRate,
            cp.Node.TargetFlow))

# update boundary conditions
for index, node in enumerate(Patient.Topology.OutletNodes):
    node.OutletFlowRate = node.WKNode.AccumulatedFlowRate * -1e-6
Patient.UpdateFlowRateCouplingPoints(-1e-9*FlowRateAtBoundary)
Patient.Results1DSteadyStateModel()
Patient.ExportMeanResults(file="ResultsPerVesselHealthy.csv")

for outlet in Patient.Topology.OutletNodes:
    outlet.Pressure = outlet.WKNode.Pressure
Patient.Perfusion.UpdateMappedRegionsFlowdata(configs.input.inlet_boundary_file)

# %% RUN COUPLED MODEL
def coupledmodel(P):
    # update boundary file and vessel outlet
    for index, node in enumerate(Patient.Perfusion.CouplingPoints):
        node.Node.OutPressure = P[index]  # set pressure at the coupling point
        node.Node.Pressure = P[index]  # for updating boundary file
    Patient.Perfusion.UpdateMappedRegionsFlowdata(configs.input.inlet_boundary_file)

    # Run perfusion model
    with contextlib.redirect_stdout(None):
        Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
            fe_mod.alloc_fct_spaces(mesh, configs.simulation.fe_degr)  # do we need to allocate them again?

        LHS, RHS, sigma1, sigma2, sigma3, BCs = \
            fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, p, p1, p2, p3, K1, K2, K3, beta12, beta23,
                                     p_arterial, p_venous,
                                     configs.input.read_inlet_boundary, configs.input.inlet_boundary_file,
                                     configs.input.inlet_BC_type)

        # lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

        p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol)
        p1, p2, p3 = p.split()
        # compute velocities
        vel1 = project(-K1 * grad(p1), Vvel, solver_type='bicgstab', preconditioner_type='amg')
        vel2 = project(-K2 * grad(p2), Vvel, solver_type='bicgstab', preconditioner_type='amg')
        vel3 = project(-K3 * grad(p3), Vvel, solver_type='bicgstab', preconditioner_type='amg')
        ps = [p1, p2, p3]
        vels = [vel1, vel2, vel3]
        # get surface values
        fluxes, surf_p_values = suppl_fcts.surface_ave(mesh, boundaries, vels, ps)

        FlowRateAtBoundary = fluxes[:, 2][2:]* -1
        # PressureAtBoundary = surf_p_values[:, 2][2:]

        # Run 1-D bf model
        Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                      FlowRateOutlets=False, coarseCollaterals=coarseCollaterals)

        # return residuals
        flowrate1d = [Node.Node.WKNode.AccumulatedFlowRate for index, Node in enumerate(Patient.Perfusion.CouplingPoints)]
        # pressure1d = [Node.Node.WKNode.Pressure for index, Node in enumerate(Patient.Perfusion.CouplingPoints)]
        residualFlowrate = [(i * 1e-3 - j) for i, j in zip(FlowRateAtBoundary, flowrate1d)]
        # residualpressure= [(i - j) for i, j in zip(PressureAtBoundary, pressure1d)]
        # print(sum([abs(i) for i in residualFlowrate]))
        # print(sum([abs(i) for i in residualpressure]))
    return residualFlowrate

clotactive = True
if solver == "krylov":
    # Find the pressure at coupling points (identical to the surface regions) such that flowrate of the models are equal.
    print("\tRunning two-way coupling by root finding.")
    guessPressure = numpy.array([node.Node.WKNode.Pressure for node in Patient.Perfusion.CouplingPoints])  # healthy scenario
    sol = scipy.optimize.root(coupledmodel, guessPressure, method='krylov',
                              options={'disp': True, 'maxiter': 5, 'ftol': 1e-06})
    print(sol)
    for index, node in enumerate(Patient.Perfusion.CouplingPoints):
        node.Node.OutPressure = sol.x[index]
        node.Node.Pressure = sol.x[index]
    Patient.Perfusion.UpdateMappedRegionsFlowdata(configs.input.inlet_boundary_file)
else:
    # set flow rate from the perfusion model as bc for the 1-D model
    # set pressure from the 1-D model as bc for the perfusion model
    print("\tRunning two-way coupling iteratively.")
    residual = 1
    configs.input.inlet_BC_type = 'NBC'
    while residual > 1e-6:
        oldflow = copy.deepcopy(FlowRateAtBoundary)

        # Run perfusion model
        Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
            fe_mod.alloc_fct_spaces(mesh, configs.simulation.fe_degr)  # do we need to allocate them again?

        LHS, RHS, sigma1, sigma2, sigma3, BCs = \
            fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, p, p1, p2, p3, K1, K2, K3, beta12,
                                     beta23,
                                     p_arterial, p_venous,
                                     configs.input.read_inlet_boundary, configs.input.inlet_boundary_file,
                                     configs.input.inlet_BC_type)

        # lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

        p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol)
        p1, p2, p3 = p.split()
        # compute velocities
        vel1 = project(-K1 * grad(p1), Vvel, solver_type='bicgstab', preconditioner_type='amg')
        vel2 = project(-K2 * grad(p2), Vvel, solver_type='bicgstab', preconditioner_type='amg')
        vel3 = project(-K3 * grad(p3), Vvel, solver_type='bicgstab', preconditioner_type='amg')
        ps = [p1, p2, p3]
        vels = [vel1, vel2, vel3]
        # get surface values
        fluxes, surf_p_values = suppl_fcts.surface_ave(mesh, boundaries, vels, ps)

        FlowRateAtBoundary = fluxes[:, 2][2:]* -1
        # PressureAtBoundary = surf_p_values[:, 2][2:]

        # flow rate residual before update 1d model
        flowrate1d = [Node.Node.WKNode.AccumulatedFlowRate for index, Node in enumerate(Patient.Perfusion.CouplingPoints)]
        residualFlowrate = sum([(i * 1e-3 - j) * (i * 1e-3 - j) for i, j in zip(FlowRateAtBoundary, flowrate1d)])

        # update 1-D blood flow model
        Patient.UpdateFlowRateCouplingPoints(-1e-9 * FlowRateAtBoundary)
        # for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
        #     cp.Node.OutletFlowRate = -1e-9 * FlowRateAtBoundary[index]

        # run 1-D blood flow model
        with contextlib.redirect_stdout(None):
            Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                          FlowRateOutlets=True, coarseCollaterals=coarseCollaterals)

        # update boundary file
        for index, node in enumerate(Patient.Perfusion.CouplingPoints):
            node.Node.Pressure = node.Node.WKNode.Pressure
        Patient.Perfusion.UpdateMappedRegionsFlowdata(configs.input.inlet_boundary_file)

        # get values at the outlets of the 1d model to the perfusion model
        pressure1d = [Node.Node.WKNode.Pressure for index, Node in enumerate(Patient.Perfusion.CouplingPoints)]
        flowrate1d = [Node.Node.WKNode.AccumulatedFlowRate for index, Node in enumerate(Patient.Perfusion.CouplingPoints)]

        # pressure residual between the models
        residualPressure = sum([(i - j) * (i - j) for i, j in zip(PressureAtBoundary, pressure1d)])

        print("Pressure (Pa)")
        print(PressureAtBoundary)
        print(pressure1d)
        print("Flow rates (mL) ")
        print(FlowRateAtBoundary)
        print(flowrate1d)

        print(f'Pressure residual: {residualPressure}')
        print(f'Flow Rate residual: {residualFlowrate}')

        # residual = residualFlowrate
        # residual = residualPressure
    print("\tCoupling loop successful.")
    configs.input.inlet_BC_type = 'DBC'

# Run perfusion model
Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
    fe_mod.alloc_fct_spaces(mesh, configs.simulation.fe_degr)  # do we need to allocate them again?

LHS, RHS, sigma1, sigma2, sigma3, BCs = \
    fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, p, p1, p2, p3, K1, K2, K3, beta12, beta23,
                             p_arterial, p_venous,
                             configs.input.read_inlet_boundary, configs.input.inlet_boundary_file,
                             configs.input.inlet_BC_type)

# lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol)
p1, p2, p3 = p.split()
# compute velocities
vel1 = project(-K1 * grad(p1), Vvel, solver_type='bicgstab', preconditioner_type='amg')
vel2 = project(-K2 * grad(p2), Vvel, solver_type='bicgstab', preconditioner_type='amg')
vel3 = project(-K3 * grad(p3), Vvel, solver_type='bicgstab', preconditioner_type='amg')
ps = [p1, p2, p3]
vels = [vel1, vel2, vel3]
# get surface values
fluxes, surf_p_values = suppl_fcts.surface_ave(mesh, boundaries, vels, ps)

FlowRateAtBoundary = fluxes[:, 2][2:] * -1
PressureAtBoundary = surf_p_values[:, 2][2:]
Patient.UpdatePressureCouplingPoints(PressureAtBoundary)
Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True, FlowRateOutlets=False,
                              coarseCollaterals=coarseCollaterals)
# export some results
Patient.Results1DSteadyStateModel()
# export data in same format at the 1-D pulsatile model
# start point t=0
TimePoint = Results.TimePoint(0)
TimePoint.Flow = [node.FlowRate for node in Patient.Topology.Nodes]
TimePoint.Pressure = [node.Pressure for node in Patient.Topology.Nodes]
TimePoint.Radius = [node.Radius for node in Patient.Topology.Nodes]
# end point, t=duration of a single heart beat
TimePoint2 = Results.TimePoint(Patient.ModelParameters['Beat_Duration'])
TimePoint2.Flow = TimePoint.Flow
TimePoint2.Pressure = TimePoint.Pressure
TimePoint2.Radius = TimePoint.Radius
Patient.Results.TimePoints = [TimePoint, TimePoint2]
Patient.Results.ExportResults(Patient.Folders.ModellingFolder + "Results.dyn")
Patient.LoadResults("Results.dyn")
Patient.GetMeanResults()

Patient.ExportMeanResults(file="ResultsPerVesselStroke.csv")
Patient.DistributeFlowTriangles()
Patient.ExportTriangleFlowData()
Patient.Results.AddResultsPerNodeToFile(Patient.Folders.ModellingFolder + "Topology.vtp")
Patient.Results.AddResultsPerVesselToFile(Patient.Folders.ModellingFolder + "Topology.vtp")

# save optimization results and model parameters
with open(patient_folder + 'Model_values_Stroke.csv', "w") as f:
    f.write(
        "Region,Resistance,Outlet Pressure(pa),WK Pressure, Perfusion Surface Pressure(pa),Old Flow Rate,Flow Rate(mm^3/s),Perfusion Flow Rate(mm^3/s)\n")
    for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
        f.write("%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n" % (
            fluxes[:, 0][2:][index],
            cp.Node.R1,
            cp.Node.Pressure,
            cp.Node.WKNode.Pressure,
            PressureAtBoundary[index],
            cp.Node.OldFlow,
            # cp.Node.FlowRate,
            cp.Node.WKNode.AccumulatedFlowRate,
            FlowRateAtBoundary[index] * 1e-3))


## save results
vars2save = [ps, vels, Ks]
fnames = ['press','vel','K']
for idx, fname in enumerate(fnames):
    for i in range(3):
        with XDMFFile(configs.output.res_fldr+fname+str(i+1)+'.xdmf') as myfile:
            myfile.write_checkpoint(vars2save[idx][i],fname+str(i+1), 0, XDMFFile.Encoding.HDF5, False)

with XDMFFile(configs.output.res_fldr+'beta12.xdmf') as myfile:
    myfile.write_checkpoint(beta12,"beta12", 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(configs.output.res_fldr+'beta23.xdmf') as myfile:
    myfile.write_checkpoint(beta23,"beta23", 0, XDMFFile.Encoding.HDF5, False)
with XDMFFile(configs.output.res_fldr+'perfusion.xdmf') as myfile:
    myfile.write_checkpoint(perfusion,'perfusion', 0, XDMFFile.Encoding.HDF5, False)

fheader = 'FE degree, K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, beta12gm, beta23gm, gmowm_beta_rat'
dom_props = numpy.array([configs.simulation.fe_degr,K1gm_ref,K2gm_ref,K3gm_ref,gmowm_perm_rat,beta12gm,beta23gm,gmowm_beta_rat])
numpy.savetxt(configs.output.res_fldr+'dom_props.csv', [dom_props],"%d,%e,%e,%e,%e,%e,%e,%e",header=fheader)

#%%

if configs.output.comp_ave == True:
    # obtain fluxes (ID, surface area, flux1, flux2, flux3)
    fluxes, surf_p_values = suppl_fcts.surface_ave(mesh,boundaries,vels,ps)

    # obtain some characteristic values within the domain (ID, volume, average, min, max)
    vol_p_values, vol_vel_values = suppl_fcts.vol_ave(mesh,subdomains,ps,vels)

    if rank ==0:
        print(fluxes,'\n')
        print(surf_p_values,'\n')
        print(vol_p_values,'\n')
        print(vol_vel_values,'\n')

        fheader = 'surface ID, Area [mm^2], Qa [mm^3/s], Qc [mm^3/s], Qv [mm^3/s]'
        numpy.savetxt(configs.output.res_fldr+'fluxes.csv', fluxes,"%d,%e,%e,%e,%e",header=fheader)

        fheader = 'surface ID, Area [mm^2], pa [Pa], pc [Pa], pv [Pa]'
        numpy.savetxt(configs.output.res_fldr+'surf_p_values.csv', surf_p_values,"%d,%e,%e,%e,%e",header=fheader)

        fheader = 'volume ID, Volume [mm^3], pa [Pa], pc [Pa], pv [Pa]'
        numpy.savetxt(configs.output.res_fldr+'vol_p_values.csv', vol_p_values,"%e,%e,%e,%e,%e",header=fheader)

        fheader = 'volume ID, Volume [mm^3], ua [m/s], uc [m/s], uv [m/s]'
        numpy.savetxt(configs.output.res_fldr+'vol_vel_values.csv', vol_vel_values,"%d,%e,%e,%e,%e",header=fheader)

end3 = time.time()
end0 = time.time()

#%% REPORT EXECUTION TIME
if rank == 0:
    oldstdout = sys.stdout
    logfile = open(configs.output.res_fldr+"time_info.log", 'w')
    sys.stdout = logfile
    print ('Total execution time [s]; \t\t\t', end0 - start0)
    print ('Step 1: Reading input files [s]; \t\t', end1 - start1)
    print ('Step 2: Solving governing equations [s]; \t\t', end2 - start2)
    print ('Step 3: Preparing and saving output [s]; \t\t', end3 - start3)
    logfile.close()
    sys.stdout = oldstdout
    print ('Execution time: \t', end0 - start0, '[s]')
    print ('Step 1: \t\t', end1 - start1, '[s]')
    print ('Step 2: \t\t', end2 - start2, '[s]')
    print ('Step 3: \t\t', end3 - start3, '[s]')