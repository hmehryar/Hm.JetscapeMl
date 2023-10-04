#!/bin/bash
#SBATCH --job-name ds-con-shuf-cnf9_q02.5_alpha0.4
#SBATCH -q primary
#SBATCH -n 4
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gy4065@wayne.edu
#SBATCH -o ds-con-shuf-config9_q02.5_alpha0.4-output-%j.out
#SBATCH -e ds-con-shuf-config9_q02.5_alpha0.4-error-%j.err
#SBATCH -t 100:0:0
echo Converting notebook to script
jupyter nbconvert --to python jet-ml-dataset-builder-step-8-cocatenate-shuffler.ipynb

echo "Running dataset step-8-dataset-concat-shuffler for CONFIG_NUMBER 9, q0=2.5, alpha_s=0.4"
python jet-ml-dataset-builder-step-8-cocatenate-shuffler.py -d 600000 -p 9 -a 0.4 -q 2.5

