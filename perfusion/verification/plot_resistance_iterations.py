"""
Plot estimate_resistance iterations vs estimate_resistance_tol from coupled_resistance_summary.csv.
One line per nx. Rows with missing nit (-1) are skipped.

Run from the perfusion folder:
    python3 verification/plot_resistance_iterations.py
    python3 verification/plot_resistance_iterations.py --nx 16 32
    python3 verification/plot_resistance_iterations.py --study_dir verification/resistance_study
"""
import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="estimate_resistance iterations vs tolerance, one line per nx")
parser.add_argument("--study_dir", type=str, default=os.path.join(here, 'resistance_study'))
parser.add_argument("--nx", type=int, nargs='+', default=None, help="nx values to plot (default: all)")
parser.add_argument("--out", type=str, default=None,
                    help="output path without extension (default: <study_dir>/resistance_iterations_vs_tol)")
args = parser.parse_args()

summary = os.path.join(args.study_dir, 'coupled_resistance_summary.csv')
rows = list(csv.DictReader(open(summary, newline='')))
if not rows:
    raise SystemExit(f'empty or missing: {summary}')

if 'estimate_resistance_nit' not in rows[0]:
    raise SystemExit(
        'estimate_resistance_nit column not found — re-run resistance_study.py with the updated solver '
        'to populate that column')

all_nx = sorted({int(r['nx']) for r in rows})

# one nit value per (nx, tol) case; rows are repeated per outlet so deduplicate
seen = set()
series = defaultdict(list)
for r in rows:
    if args.nx is not None and int(r['nx']) not in args.nx:
        continue
    nit_raw = r.get('estimate_resistance_nit', '').strip()
    if not nit_raw or nit_raw == '-1':
        continue
    key = (int(r['nx']), float(r['estimate_resistance_tol']))
    if key in seen:
        continue
    seen.add(key)
    series[int(r['nx'])].append((float(r['estimate_resistance_tol']), int(nit_raw)))

if not series:
    raise SystemExit('no rows with estimate_resistance_nit data — '
                     're-run resistance_study.py with the updated coupled_flow_solver.py')

ink, muted, grid = '#1f1f1e', '#6b6a63', '#e4e3dc'
palette = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': muted, 'axes.labelcolor': ink,
                     'xtick.color': muted, 'ytick.color': muted})

fig, ax = plt.subplots(figsize=(7, 4.5))
for nx in sorted(series):
    pts = sorted(series[nx])
    tols, vals = zip(*pts)
    i = all_nx.index(nx) % len(palette)
    ax.plot(tols, vals, color=palette[i], lw=2, zorder=3)
    ax.plot(tols, vals, ls='none', marker=markers[i], ms=8,
            color=palette[i], mec='white', mew=1.5, label=f'nx = {nx}', zorder=4)

ax.set_xscale('log')
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0e}'))
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

all_vals = [v for pts in series.values() for _, v in pts]
ax.set_ylim(max(0, min(all_vals) - 1), max(all_vals) + 1)

# ax.set_xlabel('estimate_resistance_tol')
ax.set_xlabel('baseline coupling tolerance')
ax.set_ylabel('# estimate coupling resistance iterations')
ax.grid(True, which='major', color=grid, lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False)

out = args.out or os.path.join(args.study_dir, 'resistance_iterations_vs_tol')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150, bbox_inches='tight')
    print(f'saved: {out}.{ext}')
