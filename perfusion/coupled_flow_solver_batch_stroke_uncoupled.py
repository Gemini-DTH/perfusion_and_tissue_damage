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
# sys.path.insert(0, "/mnt/project/perfusion/verification_batch/")
import analyt_fcts
sys.path.insert(0, "/app/bloodflow/")  # should not be necessary if installed already
# sys.path.insert(0, "/mnt/project/bloodflow/")  
from Blood_Flow_1D import Patient as Patients, Results, GeneralFunctions, Constants 
import contextlib
import scipy.optimize
#import analyt_fcts
import os
import pandas as pd

# solver runs is "silent" mode
set_log_level(50)

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

# %% READ INPUT
if rank == 0:
    print('Step 1: Reading input files, initialising functions and parameters')


## add for verification
os.chdir('./verification')
with open('gen_verif_files.yaml', "r") as configfile:
        configs_gen = yaml.load(configfile, yaml.SafeLoader)
coupled_model = False if 'decouple' in configs_gen['types']['couple'] else True
folder_path = f"./{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}"
print("folder_path", folder_path)
# if configs_gen['types']['couple'] != 'decouple' or configs_gen['types']['healthy'] != 'unhealthy':
#     print("it is not decouple and unhealthy")
#     sys.exit()

if configs_gen['types']['couple'] != 'couple' or configs_gen['types']['healthy'] != 'unhealthy':
    print("it is not couple and unhealthy")
    sys.exit()
os.chdir(folder_path)
## add for verification

parser = argparse.ArgumentParser(description="perfusion computation based on multi-compartment Darcy flow model")
parser.add_argument("--config_file", help="path to configuration file (string ended with /)",
                    type=str, default='./config_coupled_solver.yaml')
parser.add_argument("--config_analyt", help="path to analytical configuration file",
                    type=str, default='./config_coupled_analyt.yaml')
parser.add_argument("--config_analyt_block", help="path to analytical (block) configuration file",
                    type=str, default='./config_coupled_analyt_block.yaml')
parser.add_argument("--res_fldr", help="path to results folder (string ended with /)",
                    type=str, default=None)
parser.add_argument("--mesh_file", help="path to mesh_file",
                    type=str, default=None)
parser.add_argument("--inlet_boundary_file", help="path to inlet_boundary_file",
                    type=str, default=None)

config_file = parser.parse_args().config_file
config_analyt = parser.parse_args().config_analyt
config_analyt_block = parser.parse_args().config_analyt_block

configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)
with open(config_analyt, "r") as myconfigfile:
        config_analyt = yaml.load(myconfigfile, yaml.SafeLoader)
with open(config_analyt_block, "r") as myconfigfile:
        config_analyt_block = yaml.load(myconfigfile, yaml.SafeLoader)

config_analyt['network']['D'][2] = round((((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt['network']['D'][1], 8)
config_analyt['network']['D'][2] = round((((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt['network']['D'][1], 8)
config_analyt['network']['D'][3] = round((((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt['network']['D'][1],8)
config_analyt['network']['L_data'][1][2] = round((((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 10, 7)
config_analyt['network']['L_data'][2][2] = round((((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 10, 7)

config_analyt_block['network']['D'][2] = round((((config_analyt_block['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt_block['network']['D'][1], 8)
config_analyt_block['network']['D'][2] = round((((config_analyt_block['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt_block['network']['D'][1], 8)
config_analyt_block['network']['D'][3] = round((((config_analyt_block['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt_block['network']['D'][1],8)
config_analyt_block['network']['L_data'][1][2] = round((((config_analyt_block['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 10, 7)
config_analyt_block['network']['L_data'][2][2] = round((((config_analyt_block['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 10, 7)

def stroke_batch(resist_type, nx, fe_degr, rel_tol_esti, rel_tol_krylov, cpld_resist, cpld_conv_crit):
    print("nx", nx)
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
    print("xvec",xvec)
    #unhealthy case
    D, D_ave, G, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw = analyt_fcts.set_up_network(config_analyt_block)
    beta, Nc, l_subdom, x, beta_sub, subdom_id, BC_type_con, BC_val_con = analyt_fcts.set_up_continuum(config_analyt_block, nx)
    print("A[1][2]", A[1][2])
    #%% construct and solve linear equation system, compute results
    A = numpy.eye(Nn+2*Nc)
    b = numpy.zeros([Nn+2*Nc])
    analyt_fcts.define_network_eq(config_analyt_block, A, b, D, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw, beta, x, Nc, subdom_id, D_ave, G)
    analyt_fcts.define_continuum_eq(config_analyt_block, A, b, beta, Nn, Nc, BC_type_con, BC_val_con, G, l_subdom, x)
    xvec_block = numpy.linalg.solve(A,b)
    print("xvec_block",xvec_block)
    print("A[1][2]", A[1][2])
    AnalyticalFlow_1 = - (xvec[1] - xvec[2]) * A[1][2] #m^3/s
    print("AnalyticalFlow_1", AnalyticalFlow_1)
   
    # physical parameters
    p_arterial_healthy, p_arterial_stroke, p_venous = float(xvec[2]), float(xvec_block[2]), configs['physical']['p_venous']
    K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
        configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], configs['physical']['K3gm_ref'], \
        configs['physical']['gmowm_perm_rat']
    beta12gm, beta23gm, gmowm_beta_rat = \
        configs['physical']['beta12gm'], configs['physical']['beta23gm'], configs['physical']['gmowm_beta_rat']
    ## 1-D blood flow model
    # patient_folder = "/".join(
    #     configs['input']['inlet_boundary_file'].split("/")[:-2]) + "/"  # assume boundary file is in bf_sim folder
    # coupled_resistance_file = patient_folder + 'bf_sim/Coupled_resistance.csv'
    # full_path = os.path.abspath(coupled_resistance_file)
    # print(f"[INFO] Reading resistance file from: {full_path}")
    # run 1-D blood flow model and update boundary file
    patient_folder = '../verification_coupled_' + resist_type   
    coupled_resistance_file = f"../decouple_healthy_{configs_gen['types']['property']}/nx/nx_{nx}_{resist_type}/rel_tol_esti_{rel_tol_esti}/rel_tol_krylov_{rel_tol_krylov}/cpld_crit_{cpld_resist}/FE_2/Coupled_resistance.csv"
    print("coupled_resistance_file", os.path.abspath(coupled_resistance_file))
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

        # Patient.UpdatePressureCouplingPoints(p_arterial_healthy)
        Patient.UpdatePressureCouplingPoints(p_arterial_healthy)

        # update boundary file
        for outlet in Patient.Topology.OutletNodes:
            outlet.Pressure = outlet.OutPressure

        Patient.Perfusion.UpdateMappedRegionsFlowdata(configs['input']['inlet_boundary_file'])
    comm.Barrier()

    try:
        compartmental_model = configs['simulation']['model_type'].lower().strip()
    except KeyError:
        compartmental_model = 'acv'

    try:
        velocity_order = configs['simulation']['vel_order']
    except KeyError:
        velocity_order = fe_degr - 1

    # read mesh
    mesh_file = configs['input']['mesh_file'].replace('verification_mesh', f"../decouple_healthy_{configs_gen['types']['property']}/verification_mesh/verification_mesh_{nx}")
    mesh, subdomains, boundaries = IO_fcts.mesh_reader(mesh_file)

    # determine fct spaces
    Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
        fe_mod.alloc_fct_spaces(mesh, fe_degr, model_type=compartmental_model,
                                vel_order=velocity_order)
    # initialise permeability tensors
    permeability_folder = configs['input']['permeability_folder'].replace('verification_mesh', f"../decouple_healthy_{configs_gen['types']['property']}/verification_mesh/verification_mesh_{nx}")
        
    K1, K2, K3 = IO_fcts.initialise_permeabilities(K1_space, K2_space, mesh, permeability_folder,
                                                model_type=compartmental_model)

    if rank == 0:
        print('\t Scaling coupling coefficients and permeability tensors')

    # set coupling coefficients
    res_fldr = configs['output']['res_fldr'].replace('verification_mesh', f"../decouple_healthy_{configs_gen['types']['property']}/verification_mesh/verification_mesh_{nx}")
    beta12, beta23 = suppl_fcts.scale_coupling_coefficients(subdomains, \
                                                            beta12gm, beta23gm, gmowm_beta_rat, \
                                                            K2_space, res_fldr ,
                                                            model_type=compartmental_model)

    K1, K2, K3 = suppl_fcts.scale_permeabilities(subdomains, K1, K2, K3, \
                                                K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat, \
                                                res_fldr , model_type=compartmental_model)

    # %% SET UP FINITE ELEMENT SOLVER AND SOLVE GOVERNING EQUATIONS
    if rank == 0:
        print('Step 2: Defining and solving governing equations')

    lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', False, False, False

    exit_program = False

    if not GeneralFunctions.is_non_zero_file(coupled_resistance_file):
        # set up finite element solver
        print("Generate coupled_resistance_file!")
        sys.exit()  

    else:
        if rank == 0:
            # read  file
            print("coupled_resistance_file")
            resistances_outlet = [i.strip('\n').split(',') for i in open(coupled_resistance_file, "r")][1:]
            full_path = os.path.abspath(coupled_resistance_file)
            print(f"[INFO] Reading resistance file from: {full_path}")

            # set resistance
            for node, line in zip(Patient.Topology.OutletNodes, resistances_outlet):
                node.R2 = float(line[1])
                node.R1 = 0

    comm.Barrier()
    exit_program = comm.bcast(exit_program, root=0)
    if exit_program:
        sys.exit()

    tol_path = f"nx/nx_{nx}/cpld_conv_crit_{cpld_conv_crit}/"
    os.makedirs(tol_path, exist_ok=True)

    iter_counter = {'count': 0}
    # %% RUN COUPLED MODEL
    def coupledmodel(P, stopp):
        iter_counter['count'] += 1
        stopp[0] = comm.bcast(stopp[0], root=0)
        # update boundary file and vessel outlet
        if rank == 0:
            for index, node in enumerate(Patient.Perfusion.CouplingPoints):
                node.Node.OutPressure = P[index]  # set pressure at the coupling point
                node.Node.Pressure = P[index]  # for updating boundary file
            Patient.Perfusion.UpdateMappedRegionsFlowdata(configs['input']['inlet_boundary_file'])
        P = comm.bcast(P, root=0)
       
        print("P", P)
        # Run perfusion model
        # with contextlib.redirect_std out(None):
        Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
            fe_mod.alloc_fct_spaces(mesh, fe_degr,
                                    model_type=compartmental_model, vel_order=velocity_order)

        LHS, RHS, sigma1, sigma2, sigma3, BCs = \
            fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3,
                                    p, p1, p2, p3, K1, K2, K3, beta12, beta23,
                                    p_arterial_healthy, p_venous,
                                    configs['input']['read_inlet_boundary'], configs['input']['inlet_boundary_file'],
                                    configs['input']['inlet_BC_type'], model_type=compartmental_model)

        p, its, elapsed = fe_mod.solve_lin_sys_2(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, 1e-12, mon_conv, init_sol,
                                model_type=compartmental_model)

        # dof_coordinates_p = Vp.tabulate_dof_coordinates()
        # p_values = p.vector().get_local()
        # for i, coord in enumerate(dof_coordinates_p):
        #     x, y, z = coord
        #     print("x, y, z, p", x, y, z, p_values[i])

        myResults = {}
        suppl_fcts.compute_my_variables(p, K1, K2, K3, beta12, beta23, p_venous, Vp, Vvel, K2_space, configs, \
                                        myResults, compartmental_model, rank, save_data=False)
        my_integr_vars = {}
        surf_int_values, surf_int_header, volu_int_values, volu_int_header = \
            suppl_fcts.compute_integral_quantities(configs, myResults, my_integr_vars, \
                                                mesh, subdomains, boundaries, rank, save_data=False)
        if fe_degr == 2:
            suppl_fcts.save_p_per(tol_path,p,K1,K2,K3,beta12,beta23,p_venous,Vp,Vvel,K2_space, fe_degr,\
                                configs,myResults,compartmental_model,rank)
        # Flow rate from the perfusion model (sign to match 1-d bf model, positive flow towards the brain)
        coupled_surface_index_start = 2 if 1 in surf_int_values[:, 0] else 1
        FlowRateAtBoundary = my_integr_vars['vel1_surfint'][2:] * -1
        FlowRateAtBoundary = numpy.append(FlowRateAtBoundary, -numpy.sum(my_integr_vars['vel1_surfint'][:]))
        # Pressure from the perfusion model
        # PressureAtBoundary = my_integr_vars['press1_surfave'][2:]

            # Run 1-D bf model
        residualFlowrate = 0
        if rank == 0:
            Patient.Run1DSteadyStateModel(model="Linear", tol=1e-7, clotactive=clotactive, PressureInlets=True,
                                        FlowRateOutlets=False, coarseCollaterals=coarseCollaterals,
                                        frictionconstant=frictionconstant, scale_resistance=False)


            # return residuals
            flowrate1d = [Node.Node.WKNode.AccumulatedFlowRate for index, Node in
                        enumerate(Patient.Perfusion.CouplingPoints)]
            # pressure1d = [Node.Node.WKNode.Pressure for index, Node in enumerate(Patient.Perfusion.CouplingPoints)]
            residualFlowrate = [(i * 1e-3 - j) for i, j in zip(FlowRateAtBoundary, flowrate1d)]
            for i, j in zip(FlowRateAtBoundary, flowrate1d):
                result = i * 1e-3 - j
                print(f'FlowRateAtBoundary: {i}, flowrate1d: {j}, result: {result}')

            # print("residualFlowrate", residualFlowrate)
        residualFlowrate = comm.bcast(residualFlowrate, root=0)
        return residualFlowrate

    clotactive = True
    # clotactive = False
    # Find the pressure at coupling points (identical to the surface regions) such that flowrate of the models are equal.
    coupled_model = configs['simulation']['coupled_model'] if 'coupled_model' in configs['simulation'] else True
    number_coupling_points = suppl_fcts.region_label_assembler(boundaries)[1] - 3

    if coupled_model:
        if rank == 0:
            print("\033[96mRunning two-way coupling\033[m")
            sys.stdout.flush()
            guessPressure = numpy.array([node.Node.WKNode.Pressure for node in Patient.Perfusion.CouplingPoints])
            guessPressure = numpy.array([9999.991695728313, 14.906980527515])
            start_time = time.time()
            print("guessPressure coupled", guessPressure)
            stop = [0]
            sol = scipy.optimize.root(coupledmodel, guessPressure, args=(stop,), method='krylov',
                                    options={'disp': False, 'maxiter': 30,
                                            'fatol': cpld_conv_crit,
                                            'jac_options': {'rdiff': 1e-7}})
            
            stop = [1]
            coupledmodel(sol.x, stop)
            end_time = time.time()
            total_elapsed = end_time - start_time
            print(f"Computation time: {total_elapsed:.4f} seconds.")
            # total_iter = sol.nit
            # print(f"Optimization finished in {total_iter} iterations.")
            # print("Number of function evaluations (nfev):", sol.nfev)
        
            print("sol.x", sol.x)
            print("Number of iterations (calls to coupledmodel):", iter_counter['count'])
            sys.stdout.flush()
        else:
            stop = [0]
            guessPressure = numpy.zeros(number_coupling_points)
            while stop[0] == 0:
                coupledmodel(guessPressure, stop)
    else:  # decoupled
        if rank == 0:
            print("\033[96mRunning decoupled model\033[m")
            sys.stdout.flush()
            # decoupled, based on previous calibration, outlet resistance is dp/Q where Q is based on the surface flow and dp is the surface pressure - venous pressure
            # Update outlet pressure based on current flow rate
            clotactive = False
            # with contextlib.redirect_stdout(None):
            Patient.Run1DSteadyStateModel(model="Linear", tol=1e-10, clotactive=clotactive, PressureInlets=True,
                                        FlowRateOutlets=False, coarseCollaterals=coarseCollaterals,
                                        frictionconstant=frictionconstant, scale_resistance=False)

            brain_resistances = [(p_arterial_healthy - p_venous) / (Node.Node.WKNode.AccumulatedFlowRate * 1e-6) for index, Node in
                                enumerate(Patient.Perfusion.CouplingPoints)]
            AccumulatedFlowRate = [(Node.Node.WKNode.AccumulatedFlowRate * 1e-6) for index, Node in
                                enumerate(Patient.Perfusion.CouplingPoints)]
            print("AccumulatedFlowRate", AccumulatedFlowRate)
            clotactive = True

            def optim(pressure):
                Patient.UpdatePressureCouplingPoints(pressure)
                with contextlib.redirect_stdout(None):
                    Pressure1 = [(Node.Node.WKNode.Pressure) for index, Node in
                                    enumerate(Patient.Perfusion.CouplingPoints)]
                    print("Pressure1", Pressure1)
                    AccumulatedFlowRate1 = [(Node.Node.WKNode.AccumulatedFlowRate * 1e-6) for index, Node in
                                    enumerate(Patient.Perfusion.CouplingPoints)]
                    print("AccumulatedFlowRate1", AccumulatedFlowRate1)
                    Patient.Run1DSteadyStateModel(model="Linear", tol=1e-10, clotactive=clotactive, PressureInlets=True,
                                                FlowRateOutlets=False, coarseCollaterals=coarseCollaterals,
                                                frictionconstant=frictionconstant, scale_resistance=False)
                    
                    
                    estimated_pressure = [brain_resistances[index] * Node.Node.WKNode.AccumulatedFlowRate * 1e-6 + p_venous for
                                        index, Node in
                                        enumerate(Patient.Perfusion.CouplingPoints)]
                    
                    diff = [Node.Node.WKNode.Pressure - estimated_pressure[index] for index, Node in
                            enumerate(Patient.Perfusion.CouplingPoints)]
                    
                    Pressure2 = [(Node.Node.WKNode.Pressure) for index, Node in
                                    enumerate(Patient.Perfusion.CouplingPoints)]
                    print("Pressure2", Pressure2)
                    AccumulatedFlowRate2 = [(Node.Node.WKNode.AccumulatedFlowRate * 1e-6) for index, Node in
                                    enumerate(Patient.Perfusion.CouplingPoints)]
                    print("AccumulatedFlowRate2", AccumulatedFlowRate2)

                    
                    print("diff", diff)
                return diff


            guessPressure = numpy.array([node.Node.WKNode.Pressure for node in Patient.Perfusion.CouplingPoints])
            # guessPressure = numpy.array([float(xvec_block[2]), float(xvec_block[3])])
            print("guessPressure decoupled", guessPressure)
            sol = scipy.optimize.root(optim, guessPressure, method='krylov',
                                    options={'disp': True, 'maxiter': 50,
                                            'fatol': cpld_conv_crit,
                                            'precond': 'ilu',
                                            'jac_options': {'rdiff': 1e-6}})

            with open(configs['input']['inlet_boundary_file'], 'w') as f:
                f.write("# region I,Q [ml/s],p [Pa],feeding artery ID,BC: p->0 or Q->1\n")
                for index, i in enumerate(Patient.Perfusion.CouplingPoints):
                    flow_rate = i.Node.WKNode.AccumulatedFlowRate
                    pressure = i.Node.WKNode.Pressure
                    boundary_type = 1 if flow_rate < 1e-6 else 0
                    f.write("%d,%.16g,%.16g,%d,%d\n" % (
                        Constants.StartClusteringIndex + index, flow_rate, pressure,
                        Constants.MajorIDdict[i.Node.MajorVesselID], boundary_type))
        configs['input']['inlet_BC_type'] = "mixed"

    comm.Barrier()

    if rank == 0:
        print('Step 3: Computing velocity fields, saving results, and extracting some field variables')

    with contextlib.redirect_stdout(None):
        Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
            fe_mod.alloc_fct_spaces(mesh, fe_degr,
                                    model_type=compartmental_model, vel_order=velocity_order)

        LHS, RHS, sigma1, sigma2, sigma3, BCs = \
            fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, p, p1, p2, p3, K1, K2, K3, beta12,
                                    beta23, p_arterial_healthy, p_venous,
                                    configs['input']['read_inlet_boundary'], configs['input']['inlet_boundary_file'],
                                    configs['input']['inlet_BC_type'], model_type=compartmental_model)
    #print("final pressure")
    # p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol,
    #                          model_type=compartmental_model)
    p, its, elapsed= fe_mod.solve_lin_sys_2(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, 1e-6,mon_conv, init_sol,
                            model_type=compartmental_model)
    
    
    
    fe_mod.error_p(tol_path, nx, p, config_analyt, xvec_block) 
    if len(xvec) == 6:
        L2_norm = fe_mod.L2Norm_2(p, config_analyt, xvec_block)
    elif len(xvec) == 10:
        L2_norm = fe_mod.L2Norm_3(p, config_analyt, xvec_block)
    # L2_norm = fe_mod.L2Norm_3(p, config_analyt, xvec_block)
    comm.Barrier()

    names = [
    "L2_norm",
    "iter",
    "elapsed",
    "total_iter",
    "total_elapsed",
    ]

    values = [
        L2_norm,
        its,
        elapsed,
        iter_counter['count'],
        total_elapsed,
    ]
    
    
    with open(tol_path + "Error_values.csv", "w") as f:
        f.write("Name,Value\n")
        for name, val in zip(names, values):
            f.write(f"{name},{val:.12g}\n")


import matplotlib.pyplot as plt
# nx_list = numpy.array([16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240])
# [ 144, 160, 176, 192, 208, 224, 240]
# nx_list = config_analyt['numerical']['nx']
# cpld_conv_crit_list = [1e-2,1e-4,1e-6,1e-8]
nx_list = [16,32,48,64,80,96,112,128,144]
nx_list = [32,64,128,144,160]
nx_list = [16,32,64,128,144,160]
nx_list = [16]
fe_deg_list = [2]
cpld_crit = 1e-6
rel_tol_esti_list = [1e-06]
rel_tol_krylov_list = [1e-6]
cpld_conv_crit_list = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]
cpld_conv_crit_list = [1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19]
cpld_conv_crit_list = [1e-6]
resist_type = 'long_5'

for nx in nx_list:
    for cpld_conv_crit in cpld_conv_crit_list:
        stroke_batch(resist_type, nx, fe_deg_list[0], rel_tol_esti_list[0], rel_tol_krylov_list[0], cpld_crit, cpld_conv_crit)
