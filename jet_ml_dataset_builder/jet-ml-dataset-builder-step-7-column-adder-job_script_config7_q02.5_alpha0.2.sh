#!/bin/bash
#SBATCH --job-name dataset-column-adder-config7_q02.5_alpha0.2
#SBATCH -q primary
#SBATCH -n 4
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gy4065@wayne.edu
#SBATCH -o dataset-column-adder-config7_q02.5_alpha0.2-output-%j.out
#SBATCH -e dataset-column-adder-config7_q02.5_alpha0.2-error-%j.err
#SBATCH -t 100:0:0
echo Converting notebook to script
jupyter nbconvert --to python jet-ml-dataset-builder-step-7-column-adder.ipynb

echo "Running dataset step-7-column-adder for CONFIG_NUMBER 7, q0=2.5, alpha_s=0.2"
python jet-ml-dataset-builder-step-7-column-adder.py -i config-07-final-state-hadrons-matter-600k.dat -d 600000 -y MMAT -o jetscape-ml-benchmark-dataset-600k-matter.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-07-final-state-hadrons/ -p 7 -a 0.2 -q 2.5
