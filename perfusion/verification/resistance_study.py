# python3 verification/resistance_study.py --nx 16 32 64 --tol 1e-5 1e-7 1e-9

"""
Coupling-resistance study for the healthy verification case (coupled_flow_solver.py).

Same flow as verify_perfusion_coupled.sh, for every nx (mesh resolution) and every estimate_resistance_tol:
    1. regenerate the verification box mesh with that nx (gen_verif_files.py)
    2. regenerate the 1-D blood-flow files in bf_sim (verification_coupled/BFOnly.py with
       --vtp_file / --anatomy_file); done for every case, since the solver rewrites bf_sim
    3. remove bf_sim/Coupled_resistance.csv, so coupled_flow_solver.py runs the healthy
       (resistance estimation) branch instead of reading old resistances
    4. run the solver with that tolerance (--solver, default coupled_flow_solver_fast.py)
    5. archive Coupled_resistance.csv, Model_values_Healthy.csv, boundary_condition_file.csv and the
       FE results under <out_dir>/nx<nx>_tol<tol>/ and append the resistances to
       <out_dir>/coupled_resistance_summary.csv

The original config files are never modified (temporary copies are written to <out_dir>).
The whole bf_sim folder is backed up first and restored at the end (BFOnly.py wipes it),
and the mesh is regenerated with the original nx from config_coupled_analyt.yaml.

Run from the perfusion folder:
    python3 verification/resistance_study.py --nx 16 32 64 --tol 1e-5 1e-7 1e-9
    python3 verification/resistance_study.py --nx 32 --tol 1e-7 --np 4
    python3 verification/resistance_study.py --nx 32 --tol 1e-7 
        
"""
import os
import sys
import csv
import shutil
import argparse
import subprocess

import yaml

here = os.path.dirname(os.path.abspath(__file__))          # .../perfusion/verification
perfusion_dir = os.path.dirname(here)                      # .../perfusion

parser = argparse.ArgumentParser(description="coupling resistance vs nx and estimate_resistance_tol (healthy case)")
parser.add_argument("--nx", type=int, nargs='+', required=True, help="list of mesh resolutions along x")
parser.add_argument("--tol", type=float, nargs='+', required=True, help="list of estimate_resistance_tol values")
parser.add_argument("--config_file", type=str, default=os.path.join(here, 'config_coupled_solver.yaml'))
parser.add_argument("--config_analyt_file", type=str, default=os.path.join(here, 'config_coupled_analyt.yaml'))
parser.add_argument("--vtp_file", type=str, default='./three_vessel_network_point_radius_half.vtp',
                    help="BFOnly.py --vtp_file (relative to verification/verification_coupled)")
parser.add_argument("--anatomy_file", type=str, default='./1-D_Anatomy/1-D_Anatomy_three_half_mmHg.txt',
                    help="BFOnly.py --anatomy_file (relative to verification/verification_coupled)")
parser.add_argument("--solver", type=str, default='coupled_flow_solver.py',
                    help="solver script in the perfusion folder (coupled_flow_solver.py = original, slow plain iteration)")
parser.add_argument("--np", type=int, default=1, help="number of MPI processes for the solver (1 = no mpirun)")
parser.add_argument("--out_dir", type=str, default=os.path.join(here, 'resistance_study'))
args = parser.parse_args()

out_dir = os.path.abspath(args.out_dir)
os.makedirs(out_dir, exist_ok=True)

with open(args.config_file, "r", encoding="utf-8") as f:
    solver_cfg = yaml.safe_load(f)
with open(args.config_analyt_file, "r", encoding="utf-8") as f:
    analyt_cfg = yaml.safe_load(f)
nx_original = analyt_cfg['numerical']['nx']

# bf_sim folder, resolved exactly as coupled_flow_solver.py does (paths relative to the perfusion folder)
bf_sim = os.path.join(perfusion_dir, os.path.dirname(solver_cfg['input']['inlet_boundary_file']))
resistance_file = os.path.join(bf_sim, 'Coupled_resistance.csv')
bc_file = os.path.join(bf_sim, 'boundary_condition_file.csv')
model_values_file = os.path.join(bf_sim, 'Model_values_Healthy.csv')

bf_only_dir = os.path.dirname(bf_sim)                      # .../verification/verification_coupled

# back up the whole bf_sim folder (BFOnly.py wipes it, the solver rewrites it);
# refreshed on every run, so it always holds the current state (e.g. after config.xml changes)
backup_dir = os.path.join(out_dir, '_bf_sim_folder_backup')
if os.path.exists(backup_dir):
    shutil.rmtree(backup_dir)
shutil.copytree(bf_sim, backup_dir)


def run(cmd, cwd, log_file):
    print('  $', ' '.join(cmd), f'(cwd: {os.path.relpath(cwd, perfusion_dir) or "."})')
    with open(log_file, 'w') as log:
        res = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(f'  FAILED (exit code {res.returncode}), see {log_file}')
    return res.returncode == 0


def write_yaml(cfg, path):
    with open(path, 'w', encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def generate_mesh(nx, log_file):
    cfg = dict(analyt_cfg, numerical=dict(analyt_cfg['numerical'], nx=nx))
    tmp = os.path.join(out_dir, f'_config_analyt_nx{nx}.yaml')
    write_yaml(cfg, tmp)
    ok = run([sys.executable, 'gen_verif_files.py', '--config_analyt_file', tmp], here, log_file)
    return ok, tmp


summary_file = os.path.join(out_dir, 'coupled_resistance_summary.csv')
new_summary = not os.path.exists(summary_file)

try:
    for nx in args.nx:
        print(f'nx = {nx}: generating mesh')
        ok, analyt_tmp = generate_mesh(nx, os.path.join(out_dir, f'_gen_mesh_nx{nx}.log'))
        if not ok:
            continue

        for tol in args.tol:
            case = f'nx{nx}_tol{tol:g}'
            case_dir = os.path.join(out_dir, case)
            os.makedirs(case_dir, exist_ok=True)
            print(f'{case}: running BFOnly.py and {args.solver}')

            # fresh 1-D blood-flow files
            ok = run([sys.executable, 'BFOnly.py', './', '--vtp_file', args.vtp_file,
                      '--anatomy_file', args.anatomy_file], bf_only_dir, os.path.join(case_dir, 'bfonly.log'))
            if not ok:
                continue

            # healthy branch: no resistance file
            if os.path.exists(resistance_file):
                os.remove(resistance_file)

            cfg = dict(solver_cfg, simulation=dict(solver_cfg['simulation'], estimate_resistance_tol=tol))
            solver_tmp = os.path.join(case_dir, 'config_coupled_solver.yaml')
            write_yaml(cfg, solver_tmp)

            cmd = [sys.executable, args.solver, '--config_file', solver_tmp,
                   '--config_analyt_file', analyt_tmp, '--res_fldr', os.path.join(case_dir, 'fe_results') + '/']
            if args.np > 1:
                cmd = ['mpirun', '--allow-run-as-root', '-n', str(args.np)] + cmd
            ok = run(cmd, perfusion_dir, os.path.join(case_dir, 'solver.log'))

            if not ok or not os.path.exists(resistance_file):
                print(f'  no Coupled_resistance.csv produced for {case}')
                continue
            for fn in (resistance_file, model_values_file, bc_file):
                if os.path.exists(fn):
                    shutil.copy2(fn, case_dir)

            with open(resistance_file) as f:
                rows = list(csv.DictReader(f))

            er_nit = -1
            er_info = os.path.join(case_dir, 'fe_results', 'estimate_resistance_info.csv')
            if os.path.exists(er_info):
                with open(er_info, newline='') as f:
                    _r = list(csv.DictReader(f))
                if _r:
                    er_nit = int(_r[0]['estimate_resistance_nit'])

            with open(summary_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if new_summary:
                    writer.writerow(['nx', 'estimate_resistance_tol', 'outlet', 'resistance', 'estimate_resistance_nit'])
                    new_summary = False
                for r in rows:
                    writer.writerow([nx, tol, r['Outlet'], r['Resistance'], er_nit])
            print('  resistances:', ', '.join(r['Resistance'] for r in rows),
                  f'  estimate_resistance_nit={er_nit}')

finally:
    # restore the original state of bf_sim and the mesh
    print('restoring original bf_sim folder and mesh (nx = %s)' % nx_original)
    shutil.rmtree(bf_sim)
    shutil.copytree(backup_dir, bf_sim)
    generate_mesh(nx_original, os.path.join(out_dir, '_gen_mesh_restore.log'))

print('summary:', summary_file)
