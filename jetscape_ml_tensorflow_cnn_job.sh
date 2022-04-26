#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name HmJetscapeMlCnn

# Submit to the GPU QoS

#SBATCH -q gpu

# Request the GPU type

#SBATCH --gres=gpu:tesla

# Total number of cores, in this example it will 1 node with 1 core each.

#SBATCH -n 32

# Request memory

#SBATCH --mem=256G

# Mail when the job begins, ends, fails, requeues

#SBATCH --mail-type=ALL

# Where to send email alerts

#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file that will be hm_jetscape_ml_cnn_output_<jobid>.out

#SBATCH -o hm_jetscape_ml_cnn_output_%j.out

# Create an error file that will be hm_jetscape_ml_cnn_error_<jobid>.out

#SBATCH -e hm_jetscape_ml_cnn_errors_%j.err

# Set maximum time limit

#SBATCH -t 3:0:0


#Converting jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to script jetscape-ml-tensorflow-cnn.ipynb

echo "Setting up python version and conda shell"
ml python/3.7

source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh


#Activating conda environment
echo "Activating conda environment"
# conda activate tensorflow_env
conda activate tensorflow_gpuenv_v2

#Running simulation
echo "Running simulation"
python jetscape-ml-tensorflow-cnn.py