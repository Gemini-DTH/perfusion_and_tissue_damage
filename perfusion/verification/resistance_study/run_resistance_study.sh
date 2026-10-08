#!/bin/bash
# Resistance study workflow — coupling resistance vs mesh resolution (nx) and estimation tolerance.
# Run all commands from the perfusion/ folder inside the Docker container.
#
# OVERVIEW
# --------
# The resistance study runs the coupled solver in its healthy (resistance-estimation) branch:
# bf_sim/Coupled_resistance.csv is deliberately absent so the solver estimates the coupling
# resistances from scratch instead of reading pre-computed values. The study sweeps over nx
# (mesh resolution) and estimate_resistance_tol to show how both affect the estimated resistances.
#
# For every (nx, tol) pair the script:
#   1. Regenerates the FE verification mesh at that nx        (gen_verif_files.py)
#   2. Regenerates the 1-D blood-flow files in bf_sim         (BFOnly.py)
#   3. Removes bf_sim/Coupled_resistance.csv                  → solver takes the healthy/estimation branch
#   4. Runs the coupled solver with estimate_resistance_tol=tol
#   5. Archives Coupled_resistance.csv, Model_values_Healthy.csv, boundary_condition_file.csv
#      and the FE results under resistance_study/nx<nx>_tol<tol>/
#   6. Appends the estimated resistances to resistance_study/coupled_resistance_summary.csv
#
# bf_sim is backed up at the start and restored (with the original nx mesh) at the end.
# The original config files are never modified (temporary copies go into resistance_study/).
#
# WHY THIS STUDY IS A PREREQUISITE FOR stroke_study.py
# -----------------------------------------------------
# stroke_study.py does not run a healthy solver pass itself. Instead it copies
#   Coupled_resistance.csv  and  Model_values_Healthy.csv
# from resistance_study/nx<nx>_tol<resistance_tol>/ into bf_sim before every stroke case.
# So this study must be run first (at least at --tol 1e-3, the default resistance_tol used
# by stroke_study.py) for every nx value you intend to use in the stroke study.

# STEP 1 — full sweep (nx = 16, 32, 64; tol = 1e-4 down to 1e-9)
# ---------------------------------------------------------------
# Produces 21 case folders (3 nx × 7 tol). Runtime depends on nx; expect several minutes per case.
#
python3 verification/resistance_study.py \
    --nx 16 32 64 \
    --tol 1e-4 1e-3 1e-5 1e-6 1e-7 1e-8 1e-9

# STEP 2 — add a single new (nx, tol) without re-running everything
# -----------------------------------------------------------------
# Already-existing case folders are overwritten only if (nx, tol) matches exactly,
# so you can safely extend the sweep one case at a time.
#
# python3 verification/resistance_study.py --nx 128 --tol 1e-7

# STEP 3 — parallel run (MPI) for large nx
# ----------------------------------------
# Use --np to run the FE solver with MPI (the 1-D BFOnly.py step is always serial).
#
# python3 verification/resistance_study.py --nx 64 --tol 1e-7 --np 4

# STEP 4 — plot results
# ---------------------
# Reads coupled_resistance_summary.csv written by the steps above.

# Coupling resistance vs estimate_resistance_tol, one line per nx:
python3 verification/plot_resistance_study.py

# Show only a subset of nx values:
# python3 verification/plot_resistance_study.py --nx 16 64

# Show relative difference to the finest (nx, tol) instead of absolute resistance:
# python3 verification/plot_resistance_study.py --relative
