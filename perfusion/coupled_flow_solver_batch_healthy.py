# TODO: clear code after testing
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

import argparse
import time

import numpy
import yaml
# %% IMPORT MODULES
# installed python3 modules
from dolfin import *

numpy.set_printoptions(linewidth=200)
# ghost mode options: 'none', 'shared_facet', 'shared_vertex'
parameters['ghost_mode'] = 'none'

# added module
import IO_fcts
import suppl_fcts
import finite_element_fcts as fe_mod

import sys
sys.path.insert(0, "/app/perfusion/verification/")
# sys.path.insert(0, "/mnt/project/perfusion/verification/")
import analyt_fcts
sys.path.insert(0, "/app/bloodflow/")
# sys.path.insert(0, "/mnt/project/bloodflow/")  
from Blood_Flow_1D import Patient as Patients, Results, GeneralFunctions, Constants 
import contextlib
import scipy.optimize
#import analyt_fcts
import os
import pandas as pd
import shutil # as to copy boundary file 

# solver runs is "silent" mode
set_log_level(50)

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

start0 = time.time()

# %% READ INPUT
if rank == 0:
    print('Step 1: Reading input files, initialising functions and parameters')
start1 = time.time()

## add for verification
os.chdir('./verification')
with open('gen_verif_files.yaml', "r") as configfile:
        configs_gen = yaml.load(configfile, yaml.SafeLoader)
coupled_model = False if 'decouple' in configs_gen['types']['couple'] else True
folder_path = f"./{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}"
if configs_gen['types']['couple'] != 'decouple' or configs_gen['types']['healthy'] != 'healthy':
    print("it is not decouple and healthy")
    sys.exit()

print("folder_path", folder_path)
os.chdir(folder_path)
## add for verification

parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--config_file", help="path to configuration file (string ended with /)",
                    type=str, default='./config_coupled_solver.yaml')
parser.add_argument("--config_analyt", help="path to analytical configuration file",
                    type=str, default='./config_coupled_analyt.yaml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                    type=str, default=None)
parser.add_argument("--mesh_file", help="path to mesh_file",
                    type=str, default=None)
parser.add_argument("--inlet_boundary_file", help="path to inlet_boundary_file",
                    type=str, default=None)

config_file = parser.parse_args().config_file
config_analyt = parser.parse_args().config_analyt

configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
with open(config_analyt, "r") as myconfigfile:
        config_analyt = yaml.load(myconfigfile, yaml.SafeLoader)
# config_analyt['continuum']['area'] = float((numpy.sum(numpy.array(config_analyt['continuum']['l_subdom']))/8)*(numpy.sum(numpy.array(config_analyt['continuum']['l_subdom']))/8))
config_analyt['network']['D'][2] = round((((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt['network']['D'][1], 8)
config_analyt['network']['D'][3] = round((((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt['network']['D'][1],8)
config_analyt['network']['L_data'][1][2] = round((((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 10, 7)
config_analyt['network']['L_data'][2][2] = round((((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 10, 7)
print("config_analyt['network']['D'][2]", config_analyt['network']['D'][2])
print("config_analyt['network']['D'][3]", config_analyt['network']['D'][3])
print("config_analyt['network']['L_data'][1][2]", config_analyt['network']['L_data'][1][2])
print("config_analyt['network']['L_data'][2][2]", config_analyt['network']['L_data'][2][2])
def healthy_batch(resist_type, nx, fe_degr, rel_tol_esti, rel_tol_krylov, cpld_crit):
    
    # print("config_analyt['continuum']['area']", config_analyt['continuum']['area'])
    # healthy case
    #%% specify 1D network and continuum problems
    D, D_ave, G, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw = analyt_fcts.set_up_network(config_analyt)
    beta, Nc, l_subdom, x, beta_sub, subdom_id, BC_type_con, BC_val_con = analyt_fcts.set_up_continuum(config_analyt, nx)

    #%% construct and solve linear equation system, compute results
    A = numpy.eye(Nn+2*Nc)
    b = numpy.zeros([Nn+2*Nc])
    analyt_fcts.define_network_eq(config_analyt, A, b, D, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw, beta, x, Nc, subdom_id, D_ave, G)
    analyt_fcts.define_continuum_eq(config_analyt, A, b, beta, Nn, Nc, BC_type_con, BC_val_con, G, l_subdom, x)
    xvec = numpy.linalg.solve(A,b)
    print("A", A)
    print("-A[1][2]", -A[1][2])
    print("-A[1][0]", -A[1][0])
    print("xvec", xvec)
    AnalyticalResistance = -1/A[1][2]/2

    # Stack A and xvec as an augmented matrix [A | x]
    augmented = numpy.column_stack((A, xvec))
    numpy.savetxt("A_and_x.csv", augmented, delimiter=",")
    AnalyticalFlow = - (xvec[0] - xvec[1]) * A[1][0]/2 #m^3/s

    # physical parameters
    p_arterial_healthy, p_venous = float(xvec[2]), configs['physical']['p_venous']
    K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
        configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], configs['physical']['K3gm_ref'], \
        configs['physical']['gmowm_perm_rat']
    beta12gm, beta23gm, gmowm_beta_rat = \
        configs['physical']['beta12gm'], configs['physical']['beta23gm'], configs['physical']['gmowm_beta_rat']

    ## 1-D blood flow model
    # patient_folder = "/".join(
    #     configs['input']['inlet_boundary_file'].split("/")[:-2]) + "/"  # assume boundary file is in bf_sim folder
    # coupled_resistance_file = patient_folder + 'bf_sim/Coupled_resistance.csv'
    # run 1-D blood flow model and update boundary file
    patient_folder = '../verification_coupled_' + resist_type
    coupled_resistance_file = 'Coupled_resistance.csv'
    
    clotactive = False

    if rank == 0:
        Patient = Patients.Patient(patient_folder)
        Patient.LoadBFSimFiles()
        Patient.LoadModelParameters("Model_parameters.txt")
        Patient.LoadClusteringMapping(Patient.Folders.ModellingFolder + "Clusters.csv")
        Patient.LoadPositions()

        # frictionconstant = Patient.ModelParameters["FRICTION_C"]  # 8 = laminar, 22 = blunt
        frictionconstant = 8
        print(f"\033[91mFriction constant set to 8 currently.\033[m")

        # coarse collaterals
        coarseCollaterals = True if Patient.ModelParameters["coarse_collaterals_number"] > 0.0 else False
        if coarseCollaterals:
            Patient.Topology.coarse_collaterals_number = Patient.ModelParameters["coarse_collaterals_number"]
            # load mapping
            Patient.Perfusion.DualGraph.LoadSurface(Patient.Folders.ModellingFolder + "DualGraph.vtp")
            # identify regions
            Patient.Perfusion.DualGraph.IdentifyNeighbouringRegions()
            for out in Patient.Topology.OutletNodes:
                out.connected_cp = []
                #print("out.OutPressures", out.OutPressure)
            for NN, cp in zip(Patient.Perfusion.DualGraph.connected_regions, Patient.Perfusion.CouplingPoints):
                cp.Node.connected_cp = [Patient.Perfusion.CouplingPoints[connected_region].Node for connected_region in NN]

        Patient.Initiate1DSteadyStateModel()  # run with original wk elements
        # rigid walls
        # for node in Patient.Topology.Nodes:
        #     node.SetPressureAreaEquation_rigid()


        Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                    coarseCollaterals=coarseCollaterals, frictionconstant=frictionconstant,
                                    scale_resistance=False)

        # save old flowrates
        for index, node in enumerate(Patient.Topology.OutletNodes):
            node.OldFlow = node.WKNode.AccumulatedFlowRate

        Patient.UpdatePressureCouplingPoints(p_arterial_healthy)

        # update boundary file
        for outlet in Patient.Topology.OutletNodes:
            outlet.Pressure = outlet.OutPressure
        tol_path = f"nx/nx_{nx}_{resist_type}/rel_tol_esti_{rel_tol_esti}/rel_tol_krylov_{rel_tol_krylov}/cpld_crit_{cpld_crit}/FE_{fe_degr}/"
        os.makedirs(tol_path, exist_ok=True)
        Patient.Perfusion.UpdateMappedRegionsFlowdata(tol_path + configs['input']['inlet_boundary_file'])
    comm.Barrier()

    try:
        compartmental_model = configs['simulation']['model_type'].lower().strip()
    except KeyError:
        compartmental_model = 'acv'

    try:
        velocity_order = configs['simulation']['vel_order']
    except KeyError:
        velocity_order = fe_degr - 1

    #nx = configs['numerical']['nx']

    # read mesh
    mesh_file = configs['input']['mesh_file'].replace('verification_mesh', f'verification_mesh/verification_mesh_{nx}')
    # print(os.path.abspath(mesh_file))
    mesh, subdomains, boundaries = IO_fcts.mesh_reader(mesh_file)

    # determine fct spaces
    Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
        fe_mod.alloc_fct_spaces(mesh, fe_degr, model_type=compartmental_model,
                                vel_order=velocity_order)
    # initialise permeability tensors
    permeability_folder = configs['input']['permeability_folder'].replace('verification_mesh', f'verification_mesh/verification_mesh_{nx}')
        
    K1, K2, K3 = IO_fcts.initialise_permeabilities(K1_space, K2_space, mesh, permeability_folder,
                                                model_type=compartmental_model)
    # print("K1initialise_permeabilities", K1.vector().get_local())

    if rank == 0:
        print('\t Scaling coupling coefficients and permeability tensors')

    # set coupling coefficients
    res_fldr = configs['output']['res_fldr'].replace('verification_mesh', f'verification_mesh/verification_mesh_{nx}')
    
    beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains, \
                                                            beta12gm, beta23gm, gmowm_beta_rat, \
                                                            K2_space, res_fldr ,
                                                            model_type=compartmental_model)
    # print("beta12", beta12.vector().get_local())
    # print("beta23", beta23.vector().get_local())

    K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1, K2, K3, \
                                                K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, \
                                                res_fldr , model_type=compartmental_model)
    # print("K1scale_permeabilities", K1.vector().get_local())
    

    # %% SET UP FINITE ELEMENT SOLVER AND SOLVE GOVERNING EQUATIONS
    if rank == 0:
        print('Step 2: Defining and solving governing equations')

    lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', True, False, False

    exit_program = False
    absolute_path = os.path.abspath(coupled_resistance_file)
    print(f"The absolute path is: {absolute_path}")

    if os.path.exists(coupled_resistance_file):
        print(f"The file exists at: {coupled_resistance_file}")
    else:
        print(f"The file does not exist at: {coupled_resistance_file}")
    if not GeneralFunctions.is_non_zero_file(coupled_resistance_file):
        
        # set up finite element solver
        LHS, RHS, sigma1, sigma2, sigma3, BCs = \
            fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, p, p1, p2, p3, K1, K2, K3, beta12,
                                    beta23,
                                    p_arterial_healthy, p_venous, configs['input']['read_inlet_boundary'],
                                    tol_path + configs['input']['inlet_boundary_file'],
                                    configs['input']['inlet_BC_type'], model_type=compartmental_model)

        # tested iterative solvers for first order elements: gmres, cg, bicgstab
        # linear_solver_methods()
        # krylov_solver_preconditioners()
        if rank == 0:
            print('\t pressure computation')
        #p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol,
                                #model_type=compartmental_model)
       
        p, iter_krylov, elapsed_krylov = fe_mod.solve_lin_sys_2(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, rel_tol_krylov, mon_conv, init_sol,
                                model_type=compartmental_model)
        
      
        # %% COMPUTE VELOCITY FIELDS, SAVE SOLUTION, EXTRACT FIELD VARIABLES
        if rank == 0:
            print('Step 3: Computing velocity fields, saving results, and extracting some field variables')

        myResults = {}
        suppl_fcts.compute_my_variables(p, K1, K2, K3, beta12, beta23, p_venous, Vp, Vvel, K2_space, configs, \
                                        myResults, compartmental_model, rank)
        
        # suppl_fcts.save_p1_vel(nx,rel_tol_krylov,p,K1,K2,K3,beta12,beta23,p_venous,Vp,Vvel,K2_space, fe_degr,\
        #                  configs,myResults,compartmental_model,rank)
        my_integr_vars = {}

        # Assuming 'boundaries' is a MeshFunction for facets (boundary)
        # for f in range(boundaries.size()):
        #     print(f"Facet {f}, Boundary ID: {boundaries[f]}")

        surf_int_values, surf_int_header, volu_int_values, volu_int_header = \
            suppl_fcts.compute_integral_quantities(configs, myResults, my_integr_vars, \
                                                mesh, subdomains, boundaries, rank)

        # Flow rate from the perfusion model (sign to match 1-d bf model, positive flow towards the brain)
        FlowRateAtBoundary = my_integr_vars['vel1_surfint'][2:] * -1
        FlowRateAtBoundary = numpy.append(FlowRateAtBoundary, -numpy.sum(my_integr_vars['vel1_surfint'][:]))
        # print("my_integr_vars['vel1_surfint']", my_integr_vars['vel1_surfint'])
        # print("FlowRateAtBoundary1", FlowRateAtBoundary)
        # Pressure from the perfusion model
        PressureAtBoundary = my_integr_vars['press1_surfave'][2:]

        sys.stdout.flush()

        if rank == 0:
            perfusion_target = 600
            total_surface_flow = 600

            # without other outlets, set inlet flow rate if not optimized for the same value!
            # FlowRateAtBoundary in mm^3/s
            if len(Patient.Perfusion.CouplingPoints) == len(Patient.Topology.OutletNodes):
                print(f"\033[91mDetected only brain outlets, updating inlet flow\033[m")
                sys.stdout.flush()
                print(f"\tTotal inflow: {60 * FlowRateAtBoundary[-1] / 1000} mL/min")
                total_surface_flow = sum(
                    [FlowRateAtBoundary[index] * 1e-3 for index, _ in enumerate(Patient.Perfusion.CouplingPoints)]) * 60
                print(f"\ttotal surface flow: {total_surface_flow} mL/min")
                Patient.Topology.InletNodes[0][0].InletFlowRate = total_surface_flow * 1e-6 / 60  # m^3
                perfusion_target = total_surface_flow  # 600
        
            for cp in Patient.Perfusion.CouplingPoints:
                flow_rate = cp.Node.WKNode.AccumulatedFlowRate * 60  # 換算成 mL/min
                print(f"\tFlow at coupling point: {flow_rate} mL/min")

            print(
                f"\tTotal cerebral flow:{sum([cp.Node.WKNode.AccumulatedFlowRate for cp in Patient.Perfusion.CouplingPoints]) * 60} mL/min")
            print(f"\tInlet pressure:{Patient.Topology.InletNodes[0][0].Pressure} Pa")
            print(f"\tInlet flow:{Patient.Topology.InletNodes[0][0].FlowRate} mL/s")
            for index, node in enumerate(Patient.Topology.OutletNodes):
                node.OldFlow = node.WKNode.AccumulatedFlowRate

            print("Optimize 1-D blood flow model to match perfusion model.")
            sys.stdout.flush()
            correction = total_surface_flow / (sum(
                [FlowRateAtBoundary[index] * 1e-3 for index, _ in enumerate(Patient.Perfusion.CouplingPoints)]) * 60)
            print(f"\033[91m\tTotal Surface flux:{total_surface_flow / correction}, Correction factor:{correction}\033[m")
            # update boundary condition
            for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
                cp.Node.TargetFlow = FlowRateAtBoundary[
                                        index] * 1e-3  # * correction  # todo scaling to compensate for low bc resolution
                print("bcp.Node.TargetFlow", cp.Node.TargetFlow)
                # add another boundary flow
                # cp.Node.TargetFlow = -A[1][2] * (xvec[1] - xvec[2]) * 1e6
                print("acp.Node.TargetFlow", cp.Node.TargetFlow)

            def estimate_resistance(patient):
                start_esti = time.perf_counter()  
                patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                            coarseCollaterals=coarseCollaterals, frictionconstant=frictionconstant,
                                            scale_resistance=False)
                iter = 0
                # rel_tol = 1e-9
                relative_residual = 1
                
                while relative_residual > rel_tol_esti:
                    iter += 1
                    oldR = numpy.array([node.Node.R1 + node.Node.R2 for node in patient.Perfusion.CouplingPoints])
                    # print("oldR", oldR)
                    for index, cp in enumerate(patient.Perfusion.CouplingPoints):
                        # print(f"cp.Pressure : {cp.Node.Pressure}")
                        # print(f"cp.Node.OutPressure: {cp.Node.OutPressure}")
                        # print(f"cp.Node.TargetFlow: {cp.Node.TargetFlow}")
                        cp.Node.R2 = (cp.Node.Pressure - cp.Node.OutPressure) / (cp.Node.TargetFlow * 1e-6)
                        cp.Node.R1 = 0
                        # print(f"after cp.Node.R1: {cp.Node.R1}")
                        # print(f"after cp.Node.R2: {cp.Node.R2}")
                    with contextlib.redirect_stdout(None):
                        patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                                    coarseCollaterals=coarseCollaterals,
                                                    frictionconstant=frictionconstant,
                                                    scale_resistance=False)

                    relative_residual = max(
                        [abs(node.Node.R1 + node.Node.R2 - oldR[index]) / (node.Node.R1 + node.Node.R2) for index, node in
                        enumerate(patient.Perfusion.CouplingPoints)])
                    # print(f'\tMax relative residual1: {relative_residual}')
                    
                    sys.stdout.flush()
                end_esti = time.perf_counter() 
                elapsed_esti = end_esti - start_esti
                print("estimate resistance iteration", iter)
                return iter, elapsed_esti
                

            # update boundary conditions (original value)
            #add
            #Patient.UpdateOutletResistanceToPialSurface()
            # Patient.UpdatePressureCouplingPoints(Patient.ModelParameters["OUT_PRESSURE"])
            Patient.UpdatePressureCouplingPoints(p_arterial_healthy)
            # print("p_arterial_healthy", p_arterial_healthy)
            iter_esti_1, elapsed_esti_1 = estimate_resistance(Patient)
            print(
                f"\tTotal cerebral flow:{sum([cp.Node.WKNode.AccumulatedFlowRate for cp in Patient.Perfusion.CouplingPoints]) * 60} mL/min")
            print(f"\tInlet pressure:{Patient.Topology.InletNodes[0][0].Pressure} Pa")
            print(f"\tInlet flow:{Patient.Topology.InletNodes[0][0].FlowRate} mL/s")
            pressure_minimal = min([cp.Node.Pressure for cp in Patient.Perfusion.CouplingPoints])
            print(f"\tMinimal pressure found:{pressure_minimal}")
            iter_negative_resist = 0
            iter_esti_2 = 0
            elapsed_esti_2 = 0
            start_negative_resist = time.perf_counter()
            if pressure_minimal < p_arterial_healthy:
                iter_negative_resist += 1
                print("\033[91mSurface pressure is higher than expected! Optimizing inlet pressure \033[m")
                while pressure_minimal < p_arterial_healthy:
                    difference_surface_pressure = p_arterial_healthy - pressure_minimal
                    # increase inlet pressure
                    Patient.Topology.InletNodes[0][0].InletPressure += difference_surface_pressure
                    with contextlib.redirect_stdout(None):
                        Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                                    coarseCollaterals=coarseCollaterals,
                                                    frictionconstant=frictionconstant,
                                                    scale_resistance=True)
                    pressure_minimal = min([cp.Node.Pressure for cp in Patient.Perfusion.CouplingPoints])
                iter_esti_2, elapsed_esti_2 = estimate_resistance(Patient)
                print(f"\033[91mNew inlet Pressure:{Patient.Topology.InletNodes[0][0].InletPressure} \033[m")
                print(
                    f"\tTotal cerebral flow:{sum([cp.Node.WKNode.AccumulatedFlowRate for cp in Patient.Perfusion.CouplingPoints]) * 60} mL/min")
                print(f"\tInlet pressure:{Patient.Topology.InletNodes[0][0].Pressure} Pa")
                print(f"\tInlet flow:{Patient.Topology.InletNodes[0][0].FlowRate} mL/s")
            end_negative_resist = time.perf_counter() 
            elapsed_negative_resist = end_negative_resist - start_negative_resist
            print(f"Updating boundary conditions")
            sys.stdout.flush()
            Patient.UpdatePressureCouplingPoints(p_arterial_healthy)
            iter_esti_3, elapsed_esti_3 = estimate_resistance(Patient)

            pressure_diff = Patient.Topology.InletNodes[0][0].Pressure / Patient.Topology.InletNodes[0][0].InletPressure
            # print("Patient.Topology.InletNodes[0][0].Pressure", Patient.Topology.InletNodes[0][0].Pressure)
            # print("Patient.Topology.InletNodes[0][0].InletPressure", Patient.Topology.InletNodes[0][0].InletPressure)
            flow_rate_diff = 1e-6 * Patient.Topology.InletNodes[0][0].FlowRate / Patient.Topology.InletNodes[0][
                0].InletFlowRate
            # print("Patient.Topology.InletNodes[0][0].FlowRate", Patient.Topology.InletNodes[0][0].FlowRate)
            # print("Patient.Topology.InletNodes[0][0].InletFlowRate", Patient.Topology.InletNodes[0][0].InletFlowRate)
            # print("flow_rate_diff", flow_rate_diff)
            iter_diff = 0
            iter_esti_4 = 0
            elapsed_esti_4 = 0
            start_diff = time.perf_counter()
            while abs(pressure_diff - 1) > cpld_crit or abs(flow_rate_diff - 1) > cpld_crit:
                iter_diff += 1
                # update total resistance to maintain inlet boundary conditions.
                with contextlib.redirect_stdout(None):
                    Patient.Run1DSteadyStateModel(model="Linear", tol=1e-12, clotactive=clotactive, PressureInlets=True,
                                                coarseCollaterals=coarseCollaterals, frictionconstant=frictionconstant,
                                                scale_resistance=True)

                # update cerebral resistance to obtain target flow rates.
                iter_esti_4, elapsed_esti_4 = estimate_resistance(Patient)
                pressure_diff = Patient.Topology.InletNodes[0][0].Pressure / Patient.Topology.InletNodes[0][0].InletPressure
                # print("Patient.Topology.InletNodes[0][0].FlowRate", Patient.Topology.InletNodes[0][0].FlowRate)
                # print("Patient.Topology.InletNodes[0][0].InletFlowRate", Patient.Topology.InletNodes[0][0].InletFlowRate)
                flow_rate_diff = 1e-6 * Patient.Topology.InletNodes[0][0].FlowRate / Patient.Topology.InletNodes[0][
                    0].InletFlowRate
            
                print(
                    f"\tTotal cerebral flow:{sum([cp.Node.WKNode.AccumulatedFlowRate for cp in Patient.Perfusion.CouplingPoints]) * 60} mL/min")
                print(f"\tInlet pressure:{Patient.Topology.InletNodes[0][0].Pressure} Pa")
                print(f"\tInlet flow:{Patient.Topology.InletNodes[0][0].FlowRate} mL/s")
                print(f"\tPressure difference:{pressure_diff}")
                print(f"\tFlow rate difference:{flow_rate_diff}")
            end_diff = time.perf_counter()
            elapsed_diff = end_diff - start_diff
            print("iter_diff", iter_diff)

            
            print("Optimization complete.")
            # print('Execution time: \t', end_r - start_r, '[s]')

            # save optimization results and model parameters
            with open('Model_values_Healthy.csv', "w") as f:
                f.write(
                    "Region,Resistance,Outlet Pressure(pa),WK Pressure, Perfusion Surface Pressure(pa),Old Flow Rate,Flow Rate(mL/s),Perfusion Flow Rate(mL/s)\n")
                for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
                    f.write("%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n" % (
                        # fluxes[:, 0][2:][index],
                        surf_int_values[:, 0][2:][index],
                        cp.Node.R1 + cp.Node.R2,
                        cp.Node.Pressure, #Outlet Pressure
                        cp.Node.WKNode.Pressure, #WK Pressure
                        PressureAtBoundary[index], #Perfusion Surface Pressure
                        cp.Node.OldFlow,
                        cp.Node.WKNode.AccumulatedFlowRate,
                        cp.Node.TargetFlow))
            

            CouplingResistance = [node.Node.R1 + node.Node.R2 for node in Patient.Perfusion.CouplingPoints]
            for r in CouplingResistance:
                if r < 1e6:
                    print('\033[93m' + "Warning: Low coupling resistance found. R=%f \033[m" % r)
            # for index, cp in enumerate(Patient.Perfusion.CouplingPoints):
            #     print("cp.Node.Pressure", cp.Node.Pressure)
            #     print("cp.Node.AccumulatedFlowRate", cp.Node.AccumulatedFlowRate)
            #     print("cp.Node.TargetFlowww", cp.Node.TargetFlow)
            #     print("cp.Node.WKNode.Pressure", cp.Node.WKNode.Pressure)
            #     print("cp.Node.WKNode.AccumulatedFlowRate", cp.Node.WKNode.AccumulatedFlowRate)
            # update boundary conditions
            for index, node in enumerate(Patient.Topology.OutletNodes):
                node.OutletFlowRate = node.WKNode.AccumulatedFlowRate * -1e-6
            Patient.UpdateFlowRateCouplingPoints(-1e-9 * FlowRateAtBoundary)
            Patient.Results1DSteadyStateModel()
            Patient.ExportMeanResults(file="ResultsPerVesselHealthy.csv")

            for outlet in Patient.Topology.OutletNodes:
                outlet.Pressure = outlet.WKNode.Pressure
            Patient.Perfusion.UpdateMappedRegionsFlowdata(configs['input']['inlet_boundary_file'])
            
            # shutil.copy(configs['input']['inlet_boundary_file'], tol_path + configs['input']['inlet_boundary_file'])
            # export visual of collaterals
            if coarseCollaterals:
                Patient.Topology.addCollateralsToTopology(filename=Patient.Folders.ModellingFolder + "Collaterals.vtp")
            # save outlet resistance
            # with open(Patient.Folders.ModellingFolder + 'Coupled_resistance.csv', "w") as f:
            with open(tol_path + "/Coupled_resistance.csv", "w") as f:
                f.write("Outlet,Resistance\n")
                for index, cp in enumerate(Patient.Topology.OutletNodes):
                    f.write("%d,%.12g\n" % (
                        index,
                        cp.R1 + cp.R2))
        

    if rank == 0:
        print('\t pressure computation')
    #p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol,
                            #model_type=compartmental_model)
    
    p, iter_krylov, elapsed_krylov = fe_mod.solve_lin_sys_2(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, rel_tol_krylov, mon_conv, init_sol,
                            model_type=compartmental_model)
    
    if fe_degr == 2:
        suppl_fcts.save_p_per(tol_path,p,K1,K2,K3,beta12,beta23,p_venous,Vp,Vvel,K2_space, fe_degr,\
                            configs,myResults,compartmental_model,rank)
    fe_mod.error_p(tol_path, nx, p, config_analyt, xvec) 

    if len(xvec) == 6:
        L2_norm = fe_mod.L2Norm_2(p, config_analyt, xvec)
    elif len(xvec) == 10:
        L2_norm = fe_mod.L2Norm_3(p, config_analyt, xvec)


    # fe_mod.p1_p2_p3()
    
    NumericalFlow_1d = Patient.Topology.InletNodes[0][0].FlowRate * 1e-6 / 2# m^3/s
    NumericalFlow_3d = Patient.Topology.InletNodes[0][0].InletFlowRate / 2
    CouplingPointsResistance = numpy.array([node.Node.R1 + node.Node.R2 for node in Patient.Perfusion.CouplingPoints])
    FlowRelErr3d = (AnalyticalFlow - NumericalFlow_3d)/AnalyticalFlow
    AnaResistFullVas = -1/A[1][0] + 1/(-2*A[1][2])
    print("1/A[1][2]/2", 1/A[1][2]/2)
    print("CouplingPointsResistance[0]", CouplingPointsResistance[0])
    print("CouplingPointsResistance[1]", CouplingPointsResistance[1])
    NumResistFullVas = -1/A[1][0] + 1/(1/(-1/A[1][2]/2+CouplingPointsResistance[0]) + 1/(-1/A[1][2]/2+CouplingPointsResistance[1]))
    # print("AnaResistFullVas", AnaResistFullVas)
    # print("NumResistFullVas", NumResistFullVas)
    ResistFullVasRelError = (AnaResistFullVas - NumResistFullVas)/AnaResistFullVas
    # print("ResistFullVasRelError", ResistFullVasRelError)
    # print("-1/A[1][0]", -1/A[1][0])
    # print("-1/A[1][2]", -1/A[1][2])
    # print("CouplingPointsResistance[0]", CouplingPointsResistance[0])
    # print("CouplingPointsResistance[1]", CouplingPointsResistance[1])
    # print("ResistRelError", -CouplingPointsResistance[0]*A[1][0])
    # print("FlowRelErr", FlowRelErr)

    print("AnalyticalFlow", AnalyticalFlow)
    print("NumericalFlow_1d", NumericalFlow_1d)
    print("NumericalFlow_3d", NumericalFlow_3d)
    print("FlowRelErr3d", FlowRelErr3d)
    print("CouplingPointsResistance", CouplingPointsResistance[0])
    print("AnaResistFullVas", AnaResistFullVas)
    print("NumResistFullVas", NumResistFullVas)
    print("ResistFullVasRelError", ResistFullVasRelError)

    names = [
    "L2_norm",
    "AnalyticalFlow",
    "NumericalFlow_1d",
    "NumericalFlow_3d",
    "FlowRelErr3d",
    "CouplingPointsResistance",
    "AnaResistFullVas",
    "NumResistFullVas",
    "ResistFullVasRelError",
    "iter_esti_1",
    "iter_esti_2",
    "iter_esti_3",
    "iter_esti_4",
    "iter_negative_resist",
    "iter_diff",
    "iter_krylov",
    "elapsed_esti_1",
    "elapsed_esti_2",
    "elapsed_esti_3",
    "elapsed_esti_4",
    "elapsed_negative_resist",
    "elapsed_diff",
    "elapsed_krylov",
]

    values = [
        L2_norm,
        AnalyticalFlow,
        NumericalFlow_1d,
        NumericalFlow_3d,
        FlowRelErr3d,
        CouplingPointsResistance[0],
        AnaResistFullVas,
        NumResistFullVas,
        ResistFullVasRelError,
        iter_esti_1,
        iter_esti_2,
        iter_esti_3,
        iter_esti_4,
        iter_negative_resist,
        iter_diff,
        iter_krylov,
        elapsed_esti_1,
        elapsed_esti_2,
        elapsed_esti_3,
        elapsed_esti_4,
        elapsed_negative_resist,
        elapsed_diff,
        elapsed_krylov,
    ]

    with open(tol_path + "Error_values_2.csv", "w") as f:
        f.write("Name,Value\n")
        for name, val in zip(names, values):
            f.write(f"{name},{val:.12g}\n")

import matplotlib.pyplot as plt
nx_list = config_analyt['numerical']['nx']

nx_list = [16, 32, 64, 128, 144,160,176,192,208,224,240, 256]
nx_list = [48,80,96,112,144,160]
nx_list = [16,32,64,128,144,160]
# nx_list = [48,80,96,112,128,144,160]
nx_list = [16,32,64,48,80,96,112,128,144,160]
nx_list = [80,96,112,128,144,160,176,192,208,224,240]
fe_deg_list = [2]
cpld_crit = 1e-6
rel_tol_esti_list = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11]
rel_tol_esti_list = [1e-6]
rel_tol_krylov_list = [1e-3,1e-6,1e-12]

resist_type = 'long_5'

for nx in nx_list:
    for rel_tol_krylov in rel_tol_krylov_list:
    # for rel_tol_esti in rel_tol_esti_list:
        healthy_batch(resist_type, nx, fe_deg_list[0], rel_tol_esti_list[0], rel_tol_krylov, cpld_crit)