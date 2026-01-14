#!/bin/bash -l
#SBATCH -J fenics_simulation
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=12:00:00
#SBATCH -A plggemini2025-cpu
#SBATCH -p plgrid

set -e

cd $SCRATCHDIR

{% stage_in_artifact Model_parameters.txt %}

BASE_DIR="/net/pr2/projects/plgrid/plgggemini/perfusion_and_tissue_damage"
CONTAINER="$BASE_DIR/perfusion_and_tissue_damage.sif"
   

singularity exec \
    --bind "$BASE_DIR:/mnt/project" \
    --bind "$BASE_DIR/perfusion/patient_0:/mnt/inputs" \
    --bind "$BASE_DIR/perfusion/patient_0/bf_sim:/mnt/results" 
    --cleanenv \
    "$CONTAINER" \
    bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        export PYTHONPATH=\$HOME/.local/lib/python3.*/site-packages:\$PYTHONPATH
        export PYTHONPATH=/mnt/project/bloodflow/Blood_Flow_1D:$PYTHONPATH
        
	python3 -m pip install --user --no-cache-dir networkx
	python3 -m pip install --user --no-cache-dir vtk
	python3 -m pip install --user --no-cache-dir mgmetis
	python3 -m pip install --user --no-cache-dir pandas
	python3 -m pip install --user --no-cache-dir numpy==1.24.4
        
        cd /mnt/project/
        #cp -TR ./bloodflow/DataFiles/DefaultPatient_old "./perfusion/patient_0/"
        python3 ./bloodflow/Blood_Flow_1D/GenerateBloodflowFiles.py "./perfusion/patient_0/"
     
    "
