#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name jet_ml_dataset_builder_by_size

# Submit to the GPU QoS

#SBATCH -q primary

##SBATCH -q gpu

# Request the GPU type

##SBATCH --gres=gpu:tesla

# Total number of cores, in this example it will 1 node with 1 core each.

#SBATCH -n 1

#SBATCH -c 16

# Request memory

#SBATCH --mem=128G

# #SBATCH --mem-per-cpu=32

# Mail when the job begins, ends, fails, requeues

#SBATCH --mail-type=ALL

# Where to send email alerts

#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file that will be jet-jet_ml_dataset_builder_by_size_output_<jobid>.out

#SBATCH -o jet-jet_ml_dataset_builder_by_size_output_%j.out

# Create an error file that will be jet-jet_ml_dataset_builder_by_size_error_<jobid>.out

#SBATCH -e jet-jet_ml_dataset_builder_by_size_error_%j.err

# Set maximum time limit

#SBATCH -t 20:0:0


#Converting jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to python jet-jet_ml_dataset_builder_by_size.ipynb

echo "Setting up python version and conda shell"
ml python/3.7

source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh


#Activating conda environment
echo "Activating conda environment"
# conda activate tensorflow_env
conda activate tensorflow-gpu-v2.8

#Running simulation
echo "Running simulation"
python -u jet-jet_ml_dataset_builder_by_size.py