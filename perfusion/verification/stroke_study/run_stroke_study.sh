#!/bin/bash
# Stroke study workflow — L2 error of the coupled solver vs mesh resolution (nx) and coupling tolerance.
# Run all commands from the perfusion/ folder inside the Docker container.
#
# OVERVIEW
# --------
# The stroke study compares the coupled FE solver output against the analytical solution of
# a 3-region continuum model with a 1-D vascular network (three-vessel setup, half-mmHg units).
# For every (nx, tol) pair it runs the stroke solver, computes L2 errors for pressure and
# perfusion, and appends a row to:
#   stroke_study/stroke_study_summary.csv          (all metrics, one row per case)
#   stroke_study/L2_error_collateral_FE2.csv       (nx / collateral / L2 for the collateral plot)
#
# STEP 0 — prerequisite: resistance study (healthy state)
# -------------------------------------------------------
# stroke_study.py does NOT run the healthy solver itself. It copies
#   Coupled_resistance.csv  and  Model_values_Healthy.csv
# from a resistance_study run into bf_sim before each stroke case.
# Run the resistance study once (with the same nx values) before stroke_study.py:
#
python3 verification/resistance_study.py \
    --nx 16 32 64 \
    --tol 1e-3 \
    --vtp_file  ./three_vessel_network_point_radius_half.vtp \
    --anatomy_file ./1-D_Anatomy/1-D_Anatomy_three_half_mmHg.txt
#
# Output: verification/resistance_study/nx<nx>_tol0.001/  (one folder per nx)
# The tolerance here only controls the resistance estimation; 1e-3 is the default used by stroke_study.py.

# STEP 1 — basic stroke study (no collateral, three-vessel, default solver)
# -------------------------------------------------------------------------
# Loops over all (nx, tol) combinations. For each:
#   1. Runs analyt_coupled_models.py  → analytical stroke solution (xvec.csv, con_data.csv)
#   2. Calls gen_verif_files.py        → regenerates the FE mesh at that nx
#   3. Calls BFOnly.py                 → regenerates 1-D blood-flow files in bf_sim
#   4. Copies healthy state from resistance_study/nx<nx>_tol0.001/ into bf_sim
#   5. Runs the coupled solver         → stroke FE solution
#   6. Computes L2 errors              → appended to the two summary CSVs
# bf_sim is backed up at the start and restored (with the original nx mesh) at the end.
#
python3 verification/stroke_study.py \
    --nx 16 32 64 \
    --tol 1e-7 1e-8 1e-9 1e-10 1e-11

# STEP 2 — collateral study (vary pial collateral resistance)
# -----------------------------------------------------------
# Pass --collateral_resistance <value> [Pa s m^-3] to add a pial collateral between the two
# CBC coupling nodes. Results land in stroke_study/Rcoll_<value>/ (separate from Rcoll_0).
# Run once per resistance value; repeat for as many values as needed.
#
python3 verification/stroke_study.py \
    --nx 32 \
    --tol 1e-7 1e-8 1e-9 \
    --collateral_resistance 2.631e+14

python3 verification/stroke_study.py \
    --nx 32 \
    --tol 1e-7 1e-8 1e-9 \
    --collateral_resistance 2.631e+15

python3 verification/stroke_study.py \
    --nx 32 \
    --tol 1e-7 1e-8 1e-9 \
    --collateral_resistance 2.631e+16

python3 verification/stroke_study.py \
    --nx 32 \
    --tol 1e-7 1e-8 1e-9 \
    --collateral_resistance 2.631e+17

# STEP 3 — plot results
# ---------------------
# All plotters read the summary CSVs and/or FE checkpoints written above.

# L2 error vs tolerance at fixed nx (pressure and perfusion separately):
python3 verification/plot_stroke_study.py

# Pressure and perfusion profiles along x: analytical (line) vs FE (crosses),
# for healthy / stroke-no-collateral / stroke-with-collateral at one (nx, tol):
python3 verification/plot_profiles.py \
    --nx 32 --tol 1e-6 --rcoll 3.946e+15

# L2 error vs collateral resistance at fixed (nx, tol):
python3 verification/plot_collateral_study.py

# L2 error vs collateral resistance normalised by one vessel resistance:
python3 verification/plot_collateral_vs_R.py
