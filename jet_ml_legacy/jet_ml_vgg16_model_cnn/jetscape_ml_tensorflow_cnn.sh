#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name ml_cnn_config_05_1200k

# Submit to the GPU QoS

# #SBATCH -q primary

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

# Create an output file that will be ml_cnn_config_05_output_<jobid>.out

#SBATCH -o ml_cnn_config_05_output_%j.out

# Create an error file that will be ml_cnn_config_05_error_<jobid>.out

#SBATCH -e ml_cnn_config_05_error_%j.err

# Set maximum time limit

#SBATCH -t 150:0:0


#Converting jupyter notebook to python script
echo "Converting notebook to script"
# jupyter nbconvert --to python jetscape_ml_tensorflow_cnn.ipynb

echo "Setting up python version and conda shell"
ml python/3.7

source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh


#Activating conda environment
echo "Activating conda environment"
# conda activate tensorflow_env
conda activate tensorflow_gpuenv_v2

#Running simulation
echo "Running simulation"
python jetscape_ml_tensorflow_cnn.py