#!/bin/bash
#SBATCH --job-name ds-con-shuf-config1_q01.5_alpha0.2
#SBATCH -q primary
#SBATCH -n 4
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gy4065@wayne.edu
#SBATCH -o dataset-concat-shuffler-config1_q01.5_alpha0.2-output-%j.out
#SBATCH -e dataset-concat-shuffler-config1_q01.5_alpha0.2-error-%j.err
#SBATCH -t 100:0:0
echo Converting notebook to script
jupyter nbconvert --to python jet-ml-dataset-builder-step-8-cocatenate-shuffler.ipynb

echo "Running dataset step-7-column-adder for CONFIG_NUMBER 1, q0=1.5, alpha_s=0.2"
python jet-ml-dataset-builder-step-8-cocatenate-shuffler.py -p 1 -a 0.2 -q 1.5

