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
import numpy as np
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
import analyt_fcts
sys.path.insert(0, "/app/bloodflow/")  # should not be necessary if installed already
from Blood_Flow_1D import Patient, Results, GeneralFunctions, Constants
import contextlib
import scipy.optimize
import analyt_fcts
import os
from matplotlib.ticker import LogFormatter
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
start1 = time.time()

## add for verification
os.chdir('./verification')
with open('gen_verif_files.yaml', "r") as configfile:
        configs_gen = yaml.load(configfile, yaml.SafeLoader)
folder_path = f"./{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}"
os.chdir(folder_path)
print("folder_path", folder_path)
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
config_file_analyt = parser.parse_args().config_analyt

configs = IO_fcts.basic_flow_config_reader_yml(config_file, parser)

with open(config_file_analyt, "r") as myconfigfile:
        config_analyt = yaml.load(myconfigfile, yaml.SafeLoader)

config_analyt['network']['D'][2] = (((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt['network']['D'][1]
config_analyt['network']['D'][3] = (((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 2 * 2 - config_analyt['network']['D'][1]
config_analyt['network']['L_data'][1][2] = (((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 10
config_analyt['network']['L_data'][2][2] = (((config_analyt['network']['D'][0]/2) ** 3) / 2) ** (1 / 3) * 10

# print("config_analyt['network']['D'][2]", config_analyt['network']['D'][2])
# print("config_analyt['network']['D'][3]", config_analyt['network']['D'][3])
# print("config_analyt['network']['L_data'][1][2]", config_analyt['network']['L_data'][1][2])
# print("config_analyt['network']['L_data'][2][2]", config_analyt['network']['L_data'][2][2])

# healthy case
#%% specify 1D network and continuum problems
D, D_ave, G, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw = analyt_fcts.set_up_network(config_analyt)
beta, Nc, l_subdom, x, beta_sub, subdom_id, BC_type_con, BC_val_con = analyt_fcts.set_up_continuum(config_analyt, config_analyt['numerical']['nx'][0])

#%% construct and solve linear equation system, compute results
A = numpy.eye(Nn+2*Nc)
b = numpy.zeros([Nn+2*Nc])
analyt_fcts.define_network_eq(config_analyt, A, b, D, Nn, L, block_loc, BC_ID_ntw, BC_type_ntw, BC_val_ntw, beta, x, Nc, subdom_id, D_ave, G)
analyt_fcts.define_continuum_eq(config_analyt, A, b, beta, Nn, Nc, BC_type_con, BC_val_con, G, l_subdom, x)
xvec = numpy.linalg.solve(A,b)
# print("xvec",xvec)

# physical parameters
p_arterial, p_venous = float(xvec[2]), configs['physical']['p_venous']
K1gm_ref, K2gm_ref, K3gm_ref, gmowm_perm_rat = \
    configs['physical']['K1gm_ref'], configs['physical']['K2gm_ref'], configs['physical']['K3gm_ref'], \
    configs['physical']['gmowm_perm_rat']
beta12gm, beta23gm, gmowm_beta_rat = \
    configs['physical']['beta12gm'], configs['physical']['beta23gm'], configs['physical']['gmowm_beta_rat']

## 1-D blood flow model
# patient_folder = "/".join(
#     configs['input']['inlet_boundary_file'].split("/")[:-2]) + "/"  # assume boundary file is in bf_sim folder
# coupled_resistance_file = patient_folder + 'bf_sim/Coupled_resistance.csv'
print(f"Current working directory: {os.getcwd()}")
# ADD FOR VERIFICATION
patient_folder = '../verification_coupled'
coupled_resistance_file = 'Coupled_resistance.csv'
# ADD FOR VERIFICATION

# run 1-D blood flow model and update boundary file
clotactive = False

if rank == 0:
    Patient = Patient.Patient(patient_folder)
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
    Patient.UpdatePressureCouplingPoints(p_arterial)
    # update boundary file
    for outlet in Patient.Topology.OutletNodes:
        outlet.Pressure = outlet.OutPressure
    Patient.Perfusion.UpdateMappedRegionsFlowdata(configs['input']['inlet_boundary_file'])
comm.Barrier()

try:
    compartmental_model = configs['simulation']['model_type'].lower().strip()
except KeyError:
    compartmental_model = 'acv'

def healthy_batch(nx, fe_degr):
    # read mesh
    mesh_folder = configs['input']['mesh_file'].replace('verification_mesh', f'verification_mesh/verification_mesh_{nx}')
    # mesh_folder = configs['input']['mesh_file'].replace('verification_mesh', f'verification_mesh_{type}/verification_mesh_{nx}')
    mesh, subdomains, boundaries = IO_fcts.mesh_reader(mesh_folder)

    # determine fct spaces
    Vp, Vvel, v_1, v_2, v_3, p, p1, p2, p3, K1_space, K2_space = \
        fe_mod.alloc_fct_spaces(mesh, fe_degr, model_type=compartmental_model,
                                vel_order=fe_degr-1)
    # initialise permeability tensors
    permeability_folder = configs['input']['permeability_folder'].replace('verification_mesh', f'verification_mesh/verification_mesh_{nx}') 
    # permeability_folder = configs['input']['permeability_folder'].replace('verification_mesh', f'verification_mesh_{type}/verification_mesh_{nx}')
    K1, K2, K3 = IO_fcts.initialise_permeabilities(K1_space, K2_space, mesh, permeability_folder,
                                                model_type=compartmental_model)

    if rank == 0:
        print('\t Scaling coupling coefficients and permeability tensors')

    # set coupling coefficients
    res_fldr = configs['output']['res_fldr'].replace('verification_mesh', f'verification_mesh/verification_mesh_{nx}')
    # res_fldr = configs['output']['res_fldr'].replace('verification_mesh', f'verification_mesh_{type}/verification_mesh_{nx}')
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

    lin_solver, precond, rtol, mon_conv, init_sol = 'bicgstab', 'amg', True, False, False

    exit_program = False
    
    if not GeneralFunctions.is_non_zero_file(coupled_resistance_file):
        # set up finite element solver
        LHS, RHS, sigma1, sigma2, sigma3, BCs = \
            fe_mod.set_up_fe_solver2(mesh, subdomains, boundaries, Vp, v_1, v_2, v_3, p, p1, p2, p3, K1, K2, K3, beta12,
                                    beta23,
                                    p_arterial, p_venous, configs['input']['read_inlet_boundary'],
                                    configs['input']['inlet_boundary_file'],
                                    configs['input']['inlet_BC_type'], model_type=compartmental_model)
        if rank == 0:
            print('\t pressure computation')

        # numerical solution and 
        p = fe_mod.solve_lin_sys(Vp, LHS, RHS, BCs, lin_solver, precond, rtol, mon_conv, init_sol,
                                model_type=compartmental_model)
        
        fe_mod.check_boundary(p_venous, p, configs['input']['read_inlet_boundary'], configs['input']['inlet_boundary_file'], configs['input']['inlet_BC_type'], Vp, boundaries)


        analyt_p, num_p, num_p1, num_p2, num_p3, L2 = fe_mod.L2Norm(fe_degr, nx, p, config_analyt, xvec)
        
        os.makedirs(f"./four_edge/", exist_ok=True)
        fe_mod.save_four_edge_pressure(analyt_p, num_p, num_p1, num_p2, num_p3, filename=f"./four_edge//four_edge_pressure_deg{fe_degr}_nx{nx}.csv")
    return L2

nx_list = config_analyt['numerical']['nx']
fe_degr_list = [1,2,3]
# 原始資料讀取
try:
    existing_df = pd.read_csv(f"rmse_results.csv", index_col=0)
except FileNotFoundError:
    existing_df = pd.DataFrame()

# 當次計算
new_df = pd.DataFrame(
    [[healthy_batch(nx, fe) for nx in nx_list] for fe in fe_degr_list],
    index=fe_degr_list,
    columns=[str(nx) for nx in nx_list]
)

# 確保欄位為字串（避免類型衝突）
existing_df.columns = existing_df.columns.astype(str)
new_df.columns = new_df.columns.astype(str)

# 使用 combine_first 以保留所有舊欄位與舊資料（更新新值）
combined_df = existing_df.combine_first(new_df)  # 舊的優先，會保留沒出現的欄位
combined_df.update(new_df)  # 將新值覆蓋到原本的表格中

# 這行是關鍵 → 將欄位依數值順序排序
combined_df = combined_df[sorted(combined_df.columns.astype(int).astype(str))]

# 儲存到 CSV（保留12位小數）
combined_df.to_csv(f"rmse_results.csv", float_format="%.12f")

# import matplotlib.pyplot as plt
# from matplotlib.pyplot import subplots
# #inverted_scale = [nx[-1] * 1/x for x in nx]
# inverted_scale = [8/x for x in nx]
# fig8, ax1 = subplots()
# colors = ['r', 'g', 'b', 'r', 'c', 'orange', 'r', 'g', 'b']
# # log-log 擬合函式與公式標示
# def fit_loglog_and_annotate(ax, x, y, color,line_idx=0, base_x=0.82, base_y=0.88, mul=0.05):
#     x_log = np.log10(x)
#     y_log = np.log10(y)
#     coeffs = np.polyfit(x_log, y_log, 1)
#     a, b = coeffs
#     print("a", a)
#     print("b", b)
#     equation = f"slope = {a:.5f}"
#     #ax.figure.text(base_x, base_y - mul * line_idx, equation, color=color, fontsize=17, ha='left')
#     print("base_x", base_x)
#     print("base_y - mul * line_idx", base_y - mul * line_idx)
#     return coeffs


# # 設置斜率為1的線
# # x_values = np.linspace(4, 10, 100)
# # y1_values = np.exp(4.8) * x_values**1.14 #4.8 #4.5
# # y2_values = np.exp(1) * x_values**2.83 #1 # 0.5
# # y3_values = np.exp(-3.5) * x_values**4.92 #-3.5 #-3.5
# #line1 = ax1.plot(x_values, y1_values, color='r', linestyle='--', linewidth=2, label="Slope = 1.14")
# #line2 = ax1.plot(x_values, y2_values, color='g', linestyle='--', linewidth=2, label="Slope = 2.83")
# #line3 = ax1.plot(x_values, y3_values, color='b', linestyle='--', linewidth=2, label="Slope = 4.92")

# for seg_idx, deg in enumerate(approx_degrees):
#     print("seg_idx", seg_idx)
#     i = seg_idx * len(nx)
#     print("i", i)
#     rmse_list_y = rmse_list[i:i+len(nx)]
#     label = f'Degree {seg_idx + 1}'
#     line4 = ax1.plot(inverted_scale, rmse_list_y, color=colors[seg_idx],linestyle='-', linewidth=2, marker='x', markersize=7, label='RMSE ' + label)
#     fit_loglog_and_annotate(ax1, inverted_scale, rmse_list_y,
#                             color=colors[seg_idx],
#                             line_idx=seg_idx,
#                             base_x=0.3, base_y=0.75,
#                             mul=0.03)
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# ax1.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
# ax1.set_xlabel('Grid Size (mm)',fontsize=17)
# ax1.set_ylabel('Pressure RMSE (mmHg)', color='k',fontsize=17)
# ax1.tick_params(axis='y', labelcolor='k')
# ax1.legend(loc='best', fontsize=12)
# ax1.tick_params(axis='both', labelsize=13) 
# fig8.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
# fig8.savefig('./verification_batch/combined_RMSE_log.png', dpi=450)