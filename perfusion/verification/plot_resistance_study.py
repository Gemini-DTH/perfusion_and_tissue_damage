"""
Plot the output of resistance_study.py: mean coupling resistance (averaged over the outlets)
versus estimate_resistance_tol, one line per nx.

Default y axis: coupling-resistance relative error (num - ana) / ana, where
    ana = analytic Poiseuille resistance of a daughter vessel (full length, from config_coupled_analyt_file)
    num = analytic resistance of the *explicitly modelled* part of that same daughter vessel
          (its length in anatomy_file, which is cut short there) + the coupling resistance that
          resistance_study.py generated for it (num accounts for the whole vessel, ana is the reference).
Both use the Poiseuille resistance R = 8 mu L / (pi r^4) with the friction constant (8, laminar)
that coupled_flow_solver.py uses, and the same mu (config_coupled_analyt_file's network.mu, which
should match BLOOD_VISC in bf_sim/Model_parameters.txt).

Reads <study_dir>/coupled_resistance_summary.csv (columns nx, estimate_resistance_tol, outlet, resistance).
Each nx keeps its colour when a subset is selected with --nx.

Run from the perfusion folder:
    python3 verification/plot_resistance_study.py
    python3 verification/plot_resistance_study.py --nx 16 64 --tol 1e-5 1e-9
    python3 verification/plot_resistance_study.py --relative   # relative difference to the finest nx at the tightest tol
"""
import os
import csv
import argparse
from collections import defaultdict

import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))


def analytic_daughter_resistance(config_analyt_file, friction=8):
    """Poiseuille resistance (and length/radius) of a CBC daughter vessel in the analytic
    network model (config_coupled_analyt.yaml). Assumes all CBC daughter vessels share the
    same length/radius, true for the symmetric verification cases."""
    with open(config_analyt_file) as f:
        cfg = yaml.safe_load(f)['network']
    mu = cfg['mu']
    cbc_ends = {node for node, btype in zip(cfg['BC_ID_ntw'], cfg['BC_type_ntw']) if btype == 'CBC'}
    daughters = [(length, diam / 2) for (_, end, length), (_, _, diam) in zip(cfg['L_data'], cfg['D_data'])
                 if end in cbc_ends]
    if not daughters:
        raise ValueError('no CBC daughter vessel found in ' + config_analyt_file)
    length, radius = daughters[0]
    if any(abs(l - length) > 1e-12 or abs(r - radius) > 1e-12 for l, r in daughters):
        print('warning: CBC daughter vessels are not all identical, using the first one:', daughters)
    return mu, length, radius, friction * mu * length / (np.pi * radius ** 4)


def explicit_daughter_length(anatomy_file, skip_parent=1):
    """Length and radius [m] of a daughter vessel as explicitly resolved in the 1-D anatomy
    file (it is cut short there); skip_parent excludes the parent/inlet vessel (first row).
    Only the vessel table at the top of the file is read (it ends at the first blank line)."""
    rows = []
    with open(anatomy_file) as f:
        next(f)  # header row (ID, Name, Length (mm), ...)
        for line in f:
            if not line.strip():
                break
            rows.append(line.strip().split('\t'))
    rows = rows[skip_parent:]
    if not rows:
        raise ValueError('no daughter vessel found in ' + anatomy_file)
    length = float(rows[0][2]) * 1e-3  # mm -> m
    radius = (float(rows[0][3]) + float(rows[0][4])) / 2 * 1e-3  # mm -> m
    if any(abs(float(r[2]) * 1e-3 - length) > 1e-12 for r in rows):
        print('warning: daughter vessels in the anatomy file are not all the same length, using the first one')
    return length, radius


parser = argparse.ArgumentParser(description="coupling resistance vs estimate_resistance_tol, one line per nx")
parser.add_argument("--study_dir", type=str, default=os.path.join(here, 'resistance_study'))
parser.add_argument("--config_analyt_file", type=str, default=os.path.join(here, 'config_coupled_analyt.yaml'),
                    help="analytic network config providing the reference (ana) daughter-vessel resistance")
parser.add_argument("--anatomy_file", type=str,
                    default=os.path.join(here, 'verification_coupled', '1-D_Anatomy', '1-D_Anatomy_three_half_mmHg.txt'),
                    help="1-D anatomy file, to get how much of the daughter vessel is explicitly resolved")
parser.add_argument("--nx", type=int, nargs='+', default=None, help="nx values to plot (default: all)")
parser.add_argument("--tol", type=float, nargs='+', default=None, help="tolerances to plot (default: all)")
parser.add_argument("--relative", action='store_true',
                    help="plot (R - R_ref)/R_ref, R_ref = largest selected nx at the smallest selected tol "
                         "(instead of the default coupling-resistance relative error vs the analytic vessel)")
parser.add_argument("--out", type=str, default=None,
                    help="output path without extension (default: <study_dir>/mean_resistance_vs_tol)")
args = parser.parse_args()

summary_file = os.path.join(args.study_dir, 'coupled_resistance_summary.csv')
out_prefix = args.out or os.path.join(
    args.study_dir,
    'mean_resistance_vs_tol_relative' if args.relative else 'coupling_resistance_error_vs_tol')

# resistances per (nx, tol); the summary is appended to, so a rerun case keeps only its last outlet set
per_case = defaultdict(dict)
with open(summary_file, newline='') as f:
    for row in csv.DictReader(f):
        per_case[(int(row['nx']), float(row['estimate_resistance_tol']))][row['outlet']] = float(row['resistance'])

all_nx = sorted({nx for nx, _ in per_case})
nx_sel = [nx for nx in all_nx if args.nx is None or nx in args.nx]
if args.nx:
    missing = sorted(set(args.nx) - set(all_nx))
    if missing:
        print('nx not in the summary:', missing)


def tol_selected(tol):
    return args.tol is None or any(np.isclose(tol, t, rtol=1e-9, atol=0) for t in args.tol)


# mean over outlets
series = {}
with open(out_prefix + '.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['nx', 'estimate_resistance_tol', 'n_outlets', 'mean_resistance', 'min_resistance', 'max_resistance'])
    for nx in nx_sel:
        tols = sorted(t for (n, t) in per_case if n == nx and tol_selected(t))
        means = []
        for t in tols:
            r = np.array(list(per_case[(nx, t)].values()))
            means.append(r.mean())
            writer.writerow([nx, t, len(r), r.mean(), r.min(), r.max()])
        if tols:
            series[nx] = (np.array(tols), np.array(means))

if not series:
    raise SystemExit('nothing to plot for the selected nx / tol')


#%% plot
ink, muted, grid = '#1f1f1e', '#6b6a63', '#e4e3dc'
# categorical slots in fixed order, keyed by position among ALL nx in the summary (colour follows nx)
palette = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': muted, 'axes.labelcolor': ink,
                     'xtick.color': muted, 'ytick.color': muted})

if args.relative:
    nx_ref = max(series)
    R_ref = series[nx_ref][1][np.argmin(series[nx_ref][0])]
    transform = lambda m: (m - R_ref)/R_ref
    ylabel = f'relative difference to nx = {nx_ref}, tol = {series[nx_ref][0].min():g}'
else:
    # ana: analytic Poiseuille resistance of the full daughter vessel (config_analyt_file)
    # num: resistance of the explicitly modelled part of that vessel (anatomy_file, cut short there)
    #      + the coupling resistance resistance_study.py generated for it (the "missing" rest of the vessel)
    mu, ana_length, ana_radius, R_ana = analytic_daughter_resistance(args.config_analyt_file)
    explicit_length, explicit_radius = explicit_daughter_length(args.anatomy_file)
    R_explicit = 8 * mu * explicit_length / (np.pi * explicit_radius ** 4)
    print(f'ana: mu={mu:g} Pa s, length={ana_length * 1e3:g} mm, radius={ana_radius * 1e3:g} mm, R_ana={R_ana:.6e} Pa s m^-3')
    print(f'num: explicit length={explicit_length * 1e3:g} mm, radius={explicit_radius * 1e3:g} mm, '
          f'R_explicit={R_explicit:.6e} Pa s m^-3 (+ generated coupling resistance)')
    transform = lambda m: (m + R_explicit - R_ana) / R_ana
    # ylabel = 'coupling resistance relative error, (num - ana) / ana'
    ylabel = 'branch resistance relative error'

fig, ax = plt.subplots(figsize=(7, 4.5))
for nx, (tols, means) in series.items():
    i = all_nx.index(nx) % len(palette)
    # nx lines often coincide: distinct marker shapes + legend identify them
    ax.plot(tols, transform(means), color=palette[i], lw=2, marker=markers[i], ms=8, mec='white', mew=1.5,
            label=f'nx = {nx}', zorder=3)

ax.set_xscale('log')
# ax.set_xlabel('estimate_resistance_tol')
ax.set_xlabel('baseline coupling tolerance')
ax.set_ylabel(ylabel)
# title = ('Mean coupling resistance vs resistance tolerance (healthy case)' if args.relative else
#          'Coupling resistance relative error vs resistance tolerance (healthy case)')
# ax.set_title(title, color=ink, loc='left', fontsize=11)
ax.grid(True, which='major', color=grid, lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)


def sci_label(v, _=None):
    """tick label as 5 x 10^-2 (mathtext) instead of 5.0e-02"""
    if v == 0:
        return '0'
    mantissa, exponent = f'{v:.1e}'.split('e')
    mantissa, exponent = float(mantissa), int(exponent)
    if exponent == 0:
        return f'{mantissa:g}'
    return rf'${mantissa:g}\times10^{{{exponent}}}$'


ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(sci_label))
ax.legend(frameon=False)

fig.savefig(out_prefix + '.png', dpi=200, bbox_inches='tight')
print('written:', out_prefix + '.png', '|', out_prefix + '.csv')
