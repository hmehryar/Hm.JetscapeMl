#!/bin/bash
#SBATCH --job-name ds-con-shuf-cnf7_q02.5_alpha0.2
#SBATCH -q primary
#SBATCH -n 4
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gy4065@wayne.edu
#SBATCH -o ds-con-shuf-config7_q02.5_alpha0.2-output-%j.out
#SBATCH -e ds-con-shuf-config7_q02.5_alpha0.2-error-%j.err
#SBATCH -t 100:0:0
echo Converting notebook to script
jupyter nbconvert --to python jet-ml-dataset-builder-step-8-cocatenate-shuffler.ipynb

echo "Running dataset step-8-dataset-concat-shuffler for CONFIG_NUMBER 7, q0=2.5, alpha_s=0.2"
python jet-ml-dataset-builder-step-8-cocatenate-shuffler.py -d 600000 -p 7 -a 0.2 -q 2.5

