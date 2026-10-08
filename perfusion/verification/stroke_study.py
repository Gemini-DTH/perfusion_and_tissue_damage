"""
L2 error of the stroke case vs nx and the coupling tolerance (coupled_flow_solver_fast.py).

Flow (same as verify_perfusion_coupled.sh, looped):
    0. analyt_coupled_models.py with the current config_coupled_analyt.yaml (set block_loc first)
       -> results/xvec.csv, results/con_data.csv (copied to <run_dir>/analytical/)
    <run_dir> = <out_dir>/Rcoll_<collateral resistance> (Rcoll_0 without collateral), so runs with different
    collateral resistances are kept apart. Results of different nx / tol are added to what is
    already there; only a rerun with the same (collateral resistance, nx, tol) replaces its own folder and rows. BLOOD_VISC in bf_sim/Model_parameters.txt is set to
    network: mu of the analytical config (the resistance study / analytical model use the same viscosity).
    for every nx:
        1. gen_verif_files.py with that nx
        2. BFOnly.py (--vtp_file / --anatomy_file) in verification_coupled
        3. no healthy solver run: copy Coupled_resistance.csv and Model_values_Healthy.csv from
           resistance_study/nx<nx>_tol0.001 into bf_sim and snapshot bf_sim (healthy state)
        pial collateral: the collateral vessel between the two coupled (CBC) nodes of the analytical network
           (e.g. L_data [2, 3, ...]) is not put in the 1-D anatomy but into the solver config
           (simulation: collateral_resistance, collateral_points), so it connects the surface pressures of
           regions 20 and 21, downstream of the healthy coupling resistances
        --collateral_resistance 0: no collateral; the edge between the two coupled nodes is also removed from the
           analytical network used here (so the analytical solution has no collateral either)
        The numerical 1-D results of every case are archived in the case folder: Results.dyn (flow, pressure, radius
           of every 1-D node), ResultsPerVesselStroke.csv (mean per vessel) and Topology.vtp (geometry + results)
        for every tol (simulation: cpld_conv_crit for coupled_flow_solver.py, cpld_conv_crit_rel for
                       coupled_flow_solver_fast.py; both are set to tol):
            4. restore the healthy bf_sim snapshot, run the solver again -> stroke case
            5. from the saved stroke press1 field compute
                 L2_error            errornorm with the exact solution interpolated 3 degrees higher (main value)
                 L2_error_rel        L2_error / ||p_exact||
                 L2_error_same_space errornorm with the exact solution in the FE space itself (= fe_mod.L2Norm_3)
                 L2_perf_same        same for the perfusion (DG0, ml/min/100ml): exact 6000*beta_i*(p_exact - p_venous)
                 L2_perf_same_rel    L2_perf_same / ||perfusion_exact||
               and store a row in <out_dir>/stroke_study_summary.csv and in
               <out_dir>/L2_error_collateral_FE<fe_degr>.csv (nx, collateral_resistance, L2_error,
               cpld_conv_crit); one row per (collateral resistance, nx, tol), all other rows are kept

The original configs are not modified (temporary copies in <run_dir>). The bf_sim folder is backed up at
the start of every run; at the end it is restored and the mesh is regenerated with the original nx. results/ keeps the analytical stroke
solution of step 0 and the last healthy run.

Run from the perfusion folder (serial):
    python3 verification/stroke_study.py --nx 16 --tol 1e-7
    python3 verification/stroke_study.py --nx 16 32 --tol 1e-7 1e-9
"""
import os
import sys
import csv
import json
import time
import re
import copy
import shutil
import argparse
import subprocess

import numpy as np
import yaml
import h5py

here = os.path.dirname(os.path.abspath(__file__))          # .../perfusion/verification
perfusion_dir = os.path.dirname(here)                      # .../perfusion
sys.path.insert(0, perfusion_dir)
def read_mesh(res_dir, name):
    from dolfin import Mesh, MeshEditor, Point
    with h5py.File(os.path.join(res_dir, f'{name}.h5'), 'r') as h5:
        coords = h5[f'{name}/{name}_0/mesh/geometry'][:]
        cells = h5[f'{name}/{name}_0/mesh/topology'][:].astype(np.uintp)
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 'tetrahedron', 3, 3)
    editor.init_vertices(len(coords))
    editor.init_cells(len(cells))
    for i, xyz in enumerate(coords):
        editor.add_vertex(i, Point(*xyz))
    for i, c in enumerate(cells):
        editor.add_cell(i, c)
    editor.close()
    return mesh


def l2_errors_perfusion(res_dir, xvec, analyt_cfg):
    from dolfin import Function, FunctionSpace, XDMFFile, interpolate, errornorm, norm, set_log_level
    import finite_element_fcts as fe_mod
    set_log_level(50)
    mesh = read_mesh(res_dir, 'perfusion')
    V0 = FunctionSpace(mesh, 'DG', 0)
    perf_h = Function(V0)
    with XDMFFile(os.path.join(res_dir, 'perfusion.xdmf')) as f_in:
        f_in.read_checkpoint(perf_h, 'perfusion', 0)
    exact = fe_mod.Exact3RegionsPerfusion(analyt_cfg['continuum']['beta'], analyt_cfg['continuum']['K'], xvec,
                                          degree=1, collateral=len(xvec) == 12)
    perf_ex = interpolate(exact, V0)
    L2_perf = errornorm(perf_ex, perf_h, 'L2')
    return L2_perf, L2_perf / norm(perf_ex, 'L2')

parser = argparse.ArgumentParser(description="stroke-case L2 error vs nx and the coupling tolerance")
parser.add_argument("--nx", type=int, nargs='+', default=[16, 32, 64])
parser.add_argument("--tol", type=float, nargs='+', default=[1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
                    help="coupling tolerances, set as cpld_conv_crit (coupled_flow_solver.py, residual [mL/s]) "
                         "and cpld_conv_crit_rel (coupled_flow_solver_fast.py, scaled residual)")
parser.add_argument("--config_file", type=str, default=os.path.join(here, 'config_coupled_solver.yaml'))
parser.add_argument("--config_analyt_file", type=str, default=os.path.join(here, 'config_coupled_analyt_stroke.yaml'))
parser.add_argument("--vtp_file", type=str, default='./three_vessel_network_point_radius_half.vtp',
                    help="BFOnly.py --vtp_file (relative to verification/verification_coupled)")
parser.add_argument("--anatomy_file", type=str, default='./1-D_Anatomy/1-D_Anatomy_three_half_mmHg.txt',
                    help="BFOnly.py --anatomy_file (relative to verification/verification_coupled)")
parser.add_argument("--solver", type=str, default='coupled_flow_solver.py',
                    help="solver script in the perfusion folder")
parser.add_argument("--out_dir", type=str, default=os.path.join(here, 'stroke_study'))
parser.add_argument("--resistance_dir", type=str, default=os.path.join(here, 'resistance_study'),
                    help="resistance study folder; its nx<nx>_tol<resistance_tol>/ provides the healthy state "
                         "(Coupled_resistance.csv, Model_values_Healthy.csv) of the stroke runs")
parser.add_argument("--resistance_tol", type=float, default=1e-3,
                    help="tolerance of the resistance study run whose Coupled_resistance.csv is used")
parser.add_argument("--collateral_resistance", type=float, default=None,
                    help="pial collateral resistance [Pa s m^-3]; default: from the collateral edge between the CBC "
                         "nodes of the analytical network (0 if it is missing or in block_loc)")
args = parser.parse_args()
args.config_analyt_file = os.path.abspath(args.config_analyt_file)   # also used by a subprocess started in another folder

out_dir = os.path.abspath(args.out_dir)
os.makedirs(out_dir, exist_ok=True)

with open(args.config_file, "r", encoding="utf-8") as f:
    solver_cfg = yaml.safe_load(f)
with open(args.config_analyt_file, "r", encoding="utf-8") as f:
    analyt_cfg = yaml.safe_load(f)
nx_original = analyt_cfg['numerical']['nx']
fe_degr = int(solver_cfg['simulation']['fe_degr'])

bf_sim = os.path.join(perfusion_dir, os.path.dirname(solver_cfg['input']['inlet_boundary_file']))
bf_only_dir = os.path.dirname(bf_sim)                      # .../verification/verification_coupled
resistance_file = os.path.join(bf_sim, 'Coupled_resistance.csv')
analyt_res = os.path.join(here, analyt_cfg['res_path'])

# fresh backup of bf_sim at the start of every run (restored at the end), so a stale backup can never
# bring back old values such as BLOOD_VISC
backup_dir = os.path.join(out_dir, '_bf_sim_folder_backup')
if os.path.exists(backup_dir):
    shutil.rmtree(backup_dir)
shutil.copytree(bf_sim, backup_dir)

summary_file = os.path.join(out_dir, 'stroke_study_summary.csv')
collateral_file = os.path.join(out_dir, f'L2_error_collateral_FE{fe_degr}.csv')

# Column name for the coupling tolerance matches the config key the chosen solver actually reads.
tol_col = 'cpld_conv_crit_rel' if 'fast' in args.solver else 'cpld_conv_crit'


def run(cmd, cwd, log_file):
    print('  $', ' '.join(cmd))
    with open(log_file, 'w') as log:
        res = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        print(f'  FAILED (exit code {res.returncode}), see {log_file}')
    return res.returncode == 0


def write_yaml(cfg, path):
    with open(path, 'w', encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def replace_dir(src, dst):
    shutil.rmtree(dst)
    shutil.copytree(src, dst)


def read_optimize_result(path):
    """OptimizeResult.txt written by the solver: key:value lines."""
    res = {}
    with open(path) as f:
        for line in f:
            key, _, value = line.partition(':')
            res[key.strip()] = value.strip()
    fun = np.array(res.get('fun', '[]').strip('[]').split(), dtype=float)
    return (res.get('success') == 'True', int(res.get('nit', -1)),
            float(np.max(np.abs(fun))) if fun.size else np.nan)


def l2_errors(res_dir, xvec):
    """Both L2 errors of the stroke press1 checkpoint in res_dir against the analytical solution."""
    from dolfin import (Function, FunctionSpace, XDMFFile, interpolate, errornorm, norm, set_log_level)
    import finite_element_fcts as fe_mod
    set_log_level(50)

    mesh = read_mesh(res_dir, 'press1')
    V = FunctionSpace(mesh, 'CG', fe_degr)
    p_h = Function(V)
    with XDMFFile(os.path.join(res_dir, 'press1.xdmf')) as f_in:
        f_in.read_checkpoint(p_h, 'press1', 0)

    exact = fe_mod.Exact3Regions(analyt_cfg['continuum']['beta'], analyt_cfg['continuum']['K'], xvec,
                                 degree=fe_degr + 3, collateral=len(xvec) == 12)
    p_ex_hi = interpolate(exact, FunctionSpace(mesh, 'CG', fe_degr + 3))
    L2 = errornorm(p_ex_hi, p_h, 'L2')
    L2_same = errornorm(interpolate(exact, V), p_h, 'L2')      # as fe_mod.L2Norm_3
    return L2, L2 / norm(p_ex_hi, 'L2'), L2_same


def _row_tol(r):
    """Return the tolerance value from a CSV row regardless of which column name was used."""
    raw = r.get('cpld_conv_crit') or r.get('cpld_conv_crit_rel')
    return float(raw) if raw else None


def upsert_row(path, row, sort_key):
    """Store row in the csv file: the row with the same (collateral resistance, nx, tolerance) is replaced,
    all other rows are kept. Tolerances are compared relatively (np.isclose with atol=0), so 1e-8 and 1e-9 differ."""
    tol_key = 'cpld_conv_crit_rel' if 'cpld_conv_crit_rel' in row else 'cpld_conv_crit'
    row_tol = float(row[tol_key])
    same = lambda r: (int(r['nx']) == row['nx']
                      and np.isclose(float(r.get('collateral_resistance') or 0), row['collateral_resistance'], rtol=1e-6, atol=0)
                      and _row_tol(r) is not None
                      and np.isclose(_row_tol(r), row_tol, rtol=1e-6, atol=0))
    rows = []
    if os.path.exists(path):
        with open(path, newline='') as f:
            rows = [r for r in csv.DictReader(f) if not same(r)]
    rows.append({k: str(v) for k, v in row.items()})
    rows.sort(key=sort_key)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_summary(row):
    upsert_row(summary_file, row, lambda r: (float(r.get('collateral_resistance') or 0), int(r['nx']),
                                             -(_row_tol(r) or 0)))


def analyt_collateral_resistance(cfg):
    """Poiseuille resistance [Pa s m^-3] of the analytical network edge between the coupled (CBC) nodes,
    same formula as analyt_fcts.set_up_network (G = pi D^4 / (32 xi L mu)); 0 if there is no such edge
    or it is blocked."""
    net = cfg['network']
    cbc_nodes = {n for n, t in zip(net['BC_ID_ntw'], net['BC_type_ntw']) if t == 'CBC'}
    if len(cbc_nodes) != 2:
        return 0.0
    L = next((l for i, j, l in net['L_data'] if {i, j} == cbc_nodes), None)
    D = next((d for i, j, d in net['D_data'] if {i, j} == cbc_nodes), None)
    blocked = any(b and set(b) == cbc_nodes for b in net['block_loc'])
    if L is None or D is None or blocked:
        return 0.0
    return 32 * net['xi'] * L * net['mu'] / (np.pi * D ** 4)


def update_collateral_csv(row):
    """nx, collateral_resistance, L2_error, cpld_conv_crit; one row per (collateral resistance, nx, tol)."""
    upsert_row(collateral_file, row, lambda r: (float(r['collateral_resistance']), int(r['nx']),
                                                -float(r['cpld_conv_crit'])))


# numerical 1-D results written to bf_sim by the stroke branch of the solver, archived with every case
oned_result_files = ('Results.dyn', 'ResultsPerVesselStroke.csv', 'Topology.vtp')


def set_blood_visc(mu):
    """BLOOD_VISC of bf_sim/Model_parameters.txt := mu (BFOnly.py and the solver read it from there)."""
    path = os.path.join(bf_sim, 'Model_parameters.txt')
    with open(path) as f:
        text = f.read()
    text, n = re.subn(r'(?m)^BLOOD_VISC=.*$', f'BLOOD_VISC={mu:g}', text)
    if n != 1:
        raise RuntimeError(f'BLOOD_VISC line not found exactly once in {path}')
    with open(path, 'w') as f:
        f.write(text)


def gen_mesh(nx, log_file):
    cfg = dict(analyt_cfg, numerical=dict(analyt_cfg['numerical'], nx=nx))
    tmp = os.path.join(run_dir, f'_config_analyt_nx{nx}.yaml')
    write_yaml(cfg, tmp)
    return run([sys.executable, 'gen_verif_files.py', '--config_analyt_file', tmp], here, log_file), tmp


R_coll = (args.collateral_resistance if args.collateral_resistance is not None
          else analyt_collateral_resistance(analyt_cfg))
print(f'pial collateral resistance R_coll = {R_coll:.6e} Pa s/m^3' if R_coll > 0 else 'no pial collateral')

# results of this collateral resistance
run_dir = os.path.join(out_dir, f'Rcoll_{R_coll:.4g}')
os.makedirs(run_dir, exist_ok=True)
print('results folder:', run_dir)

analyt_config_file = args.config_analyt_file
if args.collateral_resistance == 0:
    # no collateral: the analytical network must not contain the edge between the two coupled (CBC) nodes either
    analyt_cfg = copy.deepcopy(analyt_cfg)
    net = analyt_cfg['network']
    cbc_nodes = {n for n, t in zip(net['BC_ID_ntw'], net['BC_type_ntw']) if t == 'CBC'}
    for key in ('L_data', 'D_data'):
        net[key] = [e for e in net[key] if {e[0], e[1]} != cbc_nodes]
    analyt_config_file = os.path.join(run_dir, '_config_analyt_no_collateral.yaml')
    write_yaml(analyt_cfg, analyt_config_file)

try:
    set_blood_visc(analyt_cfg['network']['mu'])
    print(f"BLOOD_VISC set to {analyt_cfg['network']['mu']:g} (network: mu)")
    # 0. analytical stroke solution
    print('analytical solution, block_loc =', analyt_cfg['network']['block_loc'])
    if not run([sys.executable, 'analyt_coupled_models.py', '--config_file', analyt_config_file],
               here, os.path.join(run_dir, '_analytical.log')):
        sys.exit(1)
    os.makedirs(os.path.join(run_dir, 'analytical'), exist_ok=True)
    for fn in ('xvec.csv', 'con_data.csv', 'P_ntw.csv', 'Q_ntw.csv'):
        shutil.copy2(os.path.join(analyt_res, fn), os.path.join(run_dir, 'analytical'))
    xvec = np.loadtxt(os.path.join(analyt_res, 'xvec.csv'), delimiter=',')

    for nx in args.nx:
        # 1.-3. mesh, 1-D files, healthy case
        print(f'nx = {nx}: mesh, BFOnly.py, healthy case')
        ok, analyt_tmp = gen_mesh(nx, os.path.join(run_dir, f'_gen_mesh_nx{nx}.log'))
        if not ok:
            continue
        if not run([sys.executable, 'BFOnly.py', './', '--vtp_file', args.vtp_file, '--anatomy_file', args.anatomy_file],
                   bf_only_dir, os.path.join(run_dir, f'_bfonly_nx{nx}.log')):
            continue
        # healthy state taken from the resistance study (no healthy solver run)
        study_dir = os.path.join(args.resistance_dir, f'nx{nx}_tol{args.resistance_tol:g}')
        study_files = ('Coupled_resistance.csv', 'Model_values_Healthy.csv')
        missing = [fn for fn in study_files if not os.path.exists(os.path.join(study_dir, fn))]
        if missing:
            print(f'  {missing} not found in {study_dir}, skipping nx = {nx}')
            continue
        for fn in study_files:
            shutil.copy2(os.path.join(study_dir, fn), bf_sim)
        healthy_bf_sim = os.path.join(run_dir, f'_bf_sim_healthy_nx{nx}')
        if os.path.exists(healthy_bf_sim):
            shutil.rmtree(healthy_bf_sim)
        shutil.copytree(bf_sim, healthy_bf_sim)

        for tol in args.tol:
            # 4. stroke case from the healthy state
            case = f'nx{nx}_tol{tol:g}'
            case_dir = os.path.join(run_dir, case)
            os.makedirs(case_dir, exist_ok=True)
            print(f'{case}: stroke case')
            replace_dir(healthy_bf_sim, bf_sim)

            # pial collateral between coupling points 0 and 1 (regions 20 and 21)
            cfg = dict(solver_cfg, simulation=dict(solver_cfg['simulation'], cpld_conv_crit=tol, cpld_conv_crit_rel=tol,
                                                   collateral_resistance=float(R_coll), collateral_points=[0, 1]))
            solver_tmp = os.path.join(case_dir, 'config_coupled_solver.yaml')
            write_yaml(cfg, solver_tmp)
            res_dir = os.path.join(case_dir, 'fe_results')
            t0 = time.time()
            ok = run([sys.executable, args.solver, '--config_file', solver_tmp, '--config_analyt_file', analyt_tmp,
                      '--res_fldr', res_dir + '/'], perfusion_dir, os.path.join(case_dir, 'solver.log'))
            runtime = time.time() - t0
            if not ok or not os.path.exists(os.path.join(res_dir, 'press1.h5')):
                print(f'  stroke case failed for {case}')
                continue
            shutil.copy2(os.path.join(bf_only_dir, 'OptimizeResult.txt'), case_dir)
            shutil.copy2(os.path.join(bf_sim, 'Model_values_Stroke.csv'), case_dir)
            for fn in oned_result_files:                          # numerical 1-D results of this case
                if os.path.exists(os.path.join(bf_sim, fn)):
                    shutil.copy2(os.path.join(bf_sim, fn), case_dir)
                else:
                    print(f'  warning: {fn} not found in bf_sim, not archived')

            # 5. L2 errors and summary row
            converged, nit, max_res = read_optimize_result(os.path.join(case_dir, 'OptimizeResult.txt'))
            L2, L2_rel, L2_same = l2_errors(res_dir, xvec)
            L2_perf, L2_perf_rel = l2_errors_perfusion(res_dir, xvec, analyt_cfg)
            krylov_nit, krylov_n_calls = -1, -1
            krylov_info = os.path.join(res_dir, 'krylov_info.csv')
            if os.path.exists(krylov_info):
                with open(krylov_info, newline='') as _f:
                    _r = list(csv.DictReader(_f))
                if _r:
                    krylov_nit = int(_r[0]['krylov_nit'])
                    krylov_n_calls = int(_r[0]['krylov_n_calls'])
            row = {'nx': nx, 'fe_degr': fe_degr, 'block_loc': json.dumps(analyt_cfg['network']['block_loc']),
                   tol_col: tol, 'collateral_resistance': R_coll, 'L2_error': L2, 'L2_error_rel': L2_rel, 'L2_error_same_space': L2_same,
                   'L2_perf_same': L2_perf, 'L2_perf_same_rel': L2_perf_rel,
                   'converged': converged, 'iterations': nit, 'max_scaled_residual': max_res,
                   'krylov_nit': krylov_nit, 'krylov_n_calls': krylov_n_calls,
                   'runtime_s': round(runtime, 1)}
            stroke_vals = np.loadtxt(os.path.join(case_dir, 'Model_values_Stroke.csv'), delimiter=',', skiprows=1, ndmin=2)
            for r in stroke_vals:
                row[f'p_surface_region{int(r[0])}'] = r[4]
                if len(r) > 8:
                    row[f'collateral_inflow_region{int(r[0])}'] = r[8]
            update_summary(row)
            update_collateral_csv({'nx': nx, 'collateral_resistance': float(R_coll), 'L2_same': L2_same,
                                   'L2_perf_same': L2_perf, 'cpld_conv_crit': tol,
                                   'krylov_nit': krylov_nit, 'krylov_n_calls': krylov_n_calls})
            print(f'  L2 = {L2:.4e} (rel {L2_rel:.3e}, same space {L2_same:.4e}), '
                  f'perfusion L2 same space = {L2_perf:.4e} (rel {L2_perf_rel:.3e}), '
                  f'converged = {converged}, iterations = {nit}, max scaled residual = {max_res:.2e}, '
                  f'krylov_nit = {krylov_nit}, krylov_n_calls = {krylov_n_calls}')

finally:
    print('restoring original bf_sim folder and mesh (nx = %s)' % nx_original)
    replace_dir(backup_dir, bf_sim)
    gen_mesh(nx_original, os.path.join(run_dir, '_gen_mesh_restore.log'))

print('summary:', summary_file)
