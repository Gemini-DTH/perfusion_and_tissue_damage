from dolfin import *

import yaml
import time
import sys
import os
from tqdm import tqdm

import IO_fcts

# define MPI variables
comm = MPI.comm_world
rank = comm.Get_rank()

set_log_level(LogLevel.ERROR)
start0 = time.time()

# %% READ INPUT
if rank == 0:
    print('Step 1: Reading the input')

# read the .yaml file
path = './config_tissue_damage.yaml' if len(sys.argv) == 0 else sys.argv[1]
with open(path, "r") as configfile:
    configs = yaml.load(configfile, yaml.SafeLoader)

# read the mesh
mesh, subdomains, boundaries = IO_fcts.mesh_reader(configs['input']['mesh_file'])

# read perfusion results folders - the folder name should match the case
healthyfile = configs['input']['healthyfile']
strokefile = configs['input']['strokefile']
treatmentfile = configs['input']['treatmentfile']

# read the time parameters - both in hours
arrival_time, recovery_time = configs['input']['arrival_time'], configs['input']['recovery_time']

# sigmoidal function parameters for hypoxia estimation
ks1, ks2 = configs['parameter']['ks1'], configs['parameter']['ks2']

# cell death model parameters
# kf - forward rate constants
# kt - toxic production constant
# kb - toxic recycle constant
kf, kt, kb = configs['parameter']['kf'], configs['parameter']['kt'], configs['parameter']['kb']

# scale ratio between grey and white matters
perfusion_scale = configs['parameter']['perfusion_gm_wm']

# cell death threshold for core
core_threshold = configs['parameter']['core_threshold']

# dt = 60  # seconds
dt = configs["parameter"]["tissue_timestep_seconds"]
arrival_steps = int(arrival_time*3600/dt) if arrival_time > 0 else 0
recovery_steps = int((recovery_time-arrival_time)*3600/dt) if recovery_time > arrival_time else 0
total_steps = arrival_steps+recovery_steps
simulation_time = total_steps*dt/3600
if rank == 0:
    print('Step 2: Reading perfusion files')
# load previous results
time_file = configs['output']['res_fldr'] + '/time.txt'


def is_non_zero_file(fpath):
    """
    Return 1 if file exists and has data.
    :param fpath: path to file
    :return: boolean
    """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


if is_non_zero_file(time_file):
    f = open(time_file, "r")  # opening the file
    output = f.read().replace('\n', '')
    total_time = float(output)
else:
    total_time = 0.0

K2_space = FunctionSpace(mesh, "DG", 0)
# read the healthy perfusion
perfusion_healthy = Function(K2_space)
f_in = XDMFFile(healthyfile)
f_in.read_checkpoint(perfusion_healthy, 'perfusion', 0)
f_in.close()

# read the perfusion after stroke before treatment
perfusion_stroke = Function(K2_space)
f_in = XDMFFile(strokefile)
f_in.read_checkpoint(perfusion_stroke, 'perfusion', 0)
f_in.close()

# read the perfusion after treatment
perfusion_treatment = Function(K2_space)
f_in = XDMFFile(treatmentfile)
f_in.read_checkpoint(perfusion_treatment, 'perfusion', 0)
f_in.close()

if rank == 0:
    print('Step 3: Calculating the infarct fraction')

# FEM implementation
T1 = FiniteElement('DG', tetrahedron, 0)
element = MixedElement([T1, T1])
T_space = FunctionSpace(mesh, element)

u_0 = Expression(("a", "a"), degree=1, a=0)
u_n = interpolate(u_0, T_space)
T_n1, T_n2 = split(u_n)  # previous step (initial condition)

t_1, t_2 = TestFunctions(T_space)
T = TrialFunction(T_space)
T_1, T_2 = split(T)  # Dead, Toxic


class K(UserExpression):
    def __init__(self, subdomains, WM, GM, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.WM = WM
        self.GM = GM

    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == 11:
            values[0] = self.WM
        else:
            values[0] = self.GM

    def value_shape(self):
        return ()


scaling = K(subdomains=subdomains, WM=perfusion_scale, GM=1, degree=0)

tissue_space = FunctionSpace(mesh, "DG", 0)
hypoxia_estimate_fem = 1-1/(1+(exp(-(ks1*perfusion_stroke*scaling+ks2)))**2)

H2 = project(hypoxia_estimate_fem, tissue_space, solver_type='bicgstab', preconditioner_type='amg')

if configs['output']['time_series']:
    file = XDMFFile(configs['output']['res_fldr']+'H.xdmf')
    file.write(H2, 1.0)  # save function as data (smaller files, ~20MB per step)
    # checkpoint -> save solution in functionspace (large file, ~160MB per step)
    # file.write_checkpoint(H2, "Hypoxic fraction", 0, XDMFFile.Encoding.HDF5, False)
    file.close()

    dead_file = XDMFFile(configs['output']['res_fldr']+'dead.xdmf')
    toxic_file = XDMFFile(configs['output']['res_fldr']+'toxic.xdmf')
    # dead_file = File(configs['output']['res_fldr'] + 'time_series/dead.pvd')
    # toxic_file = File(configs['output']['res_fldr'] + 'time_series/toxic.pvd')

dt_const = Constant(dt)
kf_const = Constant(kf)
kt_const = Constant(kt)
kb_const = Constant(kb)

# restarting from checkpoint
tissue_health_file = configs['output']['res_fldr'] + '/infarct.xdmf'
if is_non_zero_file(tissue_health_file):
    dead = Function(K2_space)
    f_in = XDMFFile(tissue_health_file)
    f_in.read_checkpoint(dead, "dead", 0)
    f_in.close()

    toxic = Function(K2_space)
    tissue_toxic_file = configs['output']['res_fldr'] + '/toxic.xdmf'
    f_in = XDMFFile(tissue_toxic_file)
    f_in.read_checkpoint(toxic, "toxic", 0)
    f_in.close()

    # T_n1 = dead
    # T_n2 = toxic
    assign(u_n, [dead, toxic])

a = Constant(1.0) - T_n1
F = (T_1 - T_n1) * t_1 * dx - dt_const*(a * kf_const * T_n2) * t_1 * dx + \
    (T_2 - T_n2) * t_2 * dx - dt_const*(a * kt_const * H2 - kb_const * T_n2 * (Constant(1.0) - H2) * a) * t_2 * dx
a, L = lhs(F), rhs(F)
T = Function(T_space)

problem = LinearVariationalProblem(a, L, T)
solver = LinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "bicgstab"
solver.parameters["preconditioner"] = "amg"

start1 = time.time()
# arrival simulation
t = 0
for n in tqdm(range(arrival_steps)):
    t += dt

    solver.solve()
    u_n.assign(T)

    if configs['output']['time_series']:
        _u_1, _u_2 = T.split()
        dead_file.write(_u_1, total_time+t/3600)
        toxic_file.write(_u_2, total_time+t/3600)
        # dead_file << (_u_1, total_time + t / 3600)
        # toxic_file << (_u_2, total_time + t / 3600)
    # dead_file.write_checkpoint(_u_1, "dead", n, XDMFFile.Encoding.HDF5, True)
    # toxic_file.write_checkpoint(_u_2, "toxic", n, XDMFFile.Encoding.HDF5, True)

dead, toxic = T.split()
infarct = project(conditional(lt(dead, Constant(core_threshold)), Constant(0.0), Constant(1.0)), K2_space,
                  solver_type='bicgstab', preconditioner_type='amg')
core = assemble(infarct * dx) * 1e-3  # mL
if rank == 0:
    print('The core volume at the start of treatment is '+str(core)+' mL')

# # update hypoxic fraction due to treatment
if recovery_steps > 0:
    hypoxia_estimate_fem = 1-1/(1+(exp(-(ks1*perfusion_treatment*scaling+ks2)))**2)
    H2_new = project(hypoxia_estimate_fem, tissue_space, solver_type='bicgstab', preconditioner_type='amg')
    H2.assign(H2_new)

    # recovery simulation
    for n in tqdm(range(recovery_steps)):
        t += dt

        solver.solve()
        u_n.assign(T)

        if configs['output']['time_series']:
            _u_1, _u_2 = T.split()
            dead_file.write(_u_1, total_time+t/3600)
            toxic_file.write(_u_2, total_time+t/3600)
            # dead_file << (_u_1, total_time + t / 3600)
            # toxic_file << (_u_2, total_time + t / 3600)

        # dead_file.write_checkpoint(_u_1, "dead", n, XDMFFile.Encoding.HDF5, True)
        # toxic_file.write_checkpoint(_u_2, "toxic", n, XDMFFile.Encoding.HDF5, True)

dead, toxic = T.split()
end1 = time.time()

infarct = project(conditional(lt(dead, Constant(core_threshold)), Constant(0.0), Constant(1.0)), K2_space,
                  solver_type='bicgstab', preconditioner_type='amg')
core = assemble(infarct * dx) * 1e-3  # mL

# with XDMFFile(configs['output']['res_fldr']+'infarct_'+str(arrival_time)+'_'+str(recovery_time)+'.xdmf') as myfile:
#     myfile.write_checkpoint(dead,"dead", 0, XDMFFile.Encoding.HDF5, False)
#
# with XDMFFile(configs['output']['res_fldr']+'toxic_'+str(arrival_time)+'_'+str(recovery_time)+'.xdmf') as myfile:
#     myfile.write_checkpoint(toxic,"toxic", 0, XDMFFile.Encoding.HDF5, False)

with XDMFFile(configs['output']['res_fldr']+'infarct.xdmf') as myfile:
    myfile.write_checkpoint(dead, "dead", 0, XDMFFile.Encoding.HDF5, False)

with XDMFFile(configs['output']['res_fldr']+'toxic.xdmf') as myfile:
    myfile.write_checkpoint(toxic, "toxic", 0, XDMFFile.Encoding.HDF5, False)

end0 = time.time()
if rank == 0:
    if len(sys.argv) >= 2:
        # The second argument indicates the path where to write a summary of
        # outcome parameters too, this now considers only the infarct core volume.
        with open(sys.argv[2], 'w') as outfile:
            yaml.safe_dump(
                {'core-volume': core},
                outfile
            )

        with open(configs['output']['res_fldr']+"tissue_health_outcome.yml", 'a') as outfile:
            yaml.safe_dump(
                # {'core-volume'+' infarct_'+str(arrival_time)+'_'+str(recovery_time): core},
                {'core-volume' + ' infarct_' + str(total_time+simulation_time): core},
                outfile
            )

    print('The core volume is '+str(core)+' mL')
    print('Infarct computation time [s]; \t\t\t', end1 - start1)
    print('Simulation finished - Total execution time [s]; \t\t\t', end0 - start0)

if rank == 0:
    f = open(time_file, "w")
    f.write(str(total_time+simulation_time))
    f.close()
