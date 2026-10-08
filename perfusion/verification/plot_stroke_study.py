"""
Plot the output of stroke_study.py: L2 error of the stroke pressure versus the coupling
tolerance cpld_conv_crit_rel, one line per nx. Non-converged cases are drawn as hollow markers.

Reads <study_dir>/stroke_study_summary.csv. Each nx keeps its colour when a subset is selected.

Run from the perfusion folder:
    python3 verification/plot_stroke_study.py
    python3 verification/plot_stroke_study.py --metric L2_error_rel --nx 32 64
    python3 verification/plot_stroke_study.py --metric L2_error_same_space --tol 1e-7 1e-9 1e-11
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
labels = {'L2_error': r'L2 error $\|p_{exact} - p_h\|$ (exact in P$_{k+3}$)',
          'L2_error_rel': r'relative L2 error $\|p_{exact} - p_h\| / \|p_{exact}\|$',
            #   'L2_error_same_space': r'L2 error, exact interpolated in P$_k$ (L2Norm_3)'
            'L2_error_same_space': r'$L^2$-norm'
          }

parser = argparse.ArgumentParser(description="stroke L2 error vs cpld_conv_crit_rel, one line per nx")
parser.add_argument("--study_dir", type=str, default=os.path.join(here, 'stroke_study'))
parser.add_argument("--metric", type=str, default='L2_error_same_space', choices=list(labels))
parser.add_argument("--nx", type=int, nargs='+', default=None, help="nx values to plot (default: all)")
parser.add_argument("--tol", type=float, nargs='+', default=None, help="tolerances to plot (default: all)")
parser.add_argument("--linear_y", action='store_true', help="linear instead of logarithmic y axis")
parser.add_argument("--out", type=str, default=None, help="output path without extension "
                    "(default: <study_dir>/<metric>_vs_tol)")
args = parser.parse_args()

rows = list(csv.DictReader(open(os.path.join(args.study_dir, 'stroke_study_summary.csv'), newline='')))
if not rows:
    raise SystemExit('empty summary')

all_nx = sorted({int(r['nx']) for r in rows})


def selected(r):
    ok_nx = args.nx is None or int(r['nx']) in args.nx
    ok_tol = args.tol is None or any(np.isclose(float(r['cpld_conv_crit_rel']), t, rtol=1e-9, atol=0) for t in args.tol)
    return ok_nx and ok_tol


series = defaultdict(list)
for r in filter(selected, rows):
    series[int(r['nx'])].append((float(r['cpld_conv_crit_rel']), float(r[args.metric]), r['converged'] == 'True'))
if not series:
    raise SystemExit('nothing to plot for the selected nx / tol')

#%% plot
ink, muted, grid = '#1f1f1e', '#6b6a63', '#e4e3dc'
palette = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': muted, 'axes.labelcolor': ink,
                     'xtick.color': muted, 'ytick.color': muted})

fig, ax = plt.subplots(figsize=(7, 4.5))
any_failed = False
for nx in sorted(series):
    pts = sorted(series[nx])
    tols, vals, conv = (np.array(v) for v in zip(*pts))
    i = all_nx.index(nx) % len(palette)
    ax.plot(tols, vals, color=palette[i], lw=2, zorder=3)
    ax.plot(tols[conv], vals[conv], ls='none', marker=markers[i], ms=8, color=palette[i], mec='white', mew=1.5,
            label=f'nx = {nx}', zorder=4)
    if (~conv).any():
        any_failed = True
        ax.plot(tols[~conv], vals[~conv], ls='none', marker=markers[i], ms=8, mfc='white', mec=palette[i], mew=1.5,
                zorder=4)

ax.set_xscale('log')
# never zoom into round-off: the y range spans at least a factor ~2 around the data, so a flat result looks flat
all_vals = np.array([v for pts in series.values() for _, v, _ in pts])
lo, hi = all_vals.min(), all_vals.max()
if not args.linear_y:
    ax.set_yscale('log')
    if hi / lo < 2:
        mid = np.sqrt(lo * hi)
        ax.set_ylim(mid / 1.5, mid * 1.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.3g}'))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.3g}'))
elif hi - lo < 0.1 * max(abs(hi), 1e-300):
    ax.set_ylim(lo - 0.25 * abs(hi), hi + 0.25 * abs(hi))
# ax.set_xlabel('coupling tolerance cpld_conv_crit_rel')
ax.set_xlabel('stroke coupling tolerance')
ax.set_ylabel(labels[args.metric])
block = rows[0].get('block_loc', '')
# ax.set_title(f'Stroke case (block_loc {block}): L2 error vs coupling tolerance', color=ink, loc='left', fontsize=11)
ax.grid(True, which='major', color=grid, lw=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, title='hollow = not converged' if any_failed else None, title_fontsize=8)

out_prefix = args.out or os.path.join(args.study_dir, f'{args.metric}_vs_tol')
fig.savefig(out_prefix + '.png', dpi=200, bbox_inches='tight')
print('written:', out_prefix + '.png')
