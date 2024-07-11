#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name synthesis_pointnet_model_kfolded_all_datasets

# Submit to the GPU QoS

#SBATCH -q primary

##SBATCH -q gpu

# Request the GPU type

##SBATCH --gres=gpu:tesla
# #SBATCH --gres=gpu:geforc
#SBATCH --gres=gpu:2

# Request v100 gpu

##SBATCH --constraint=v100

# Total number of cores, in this example it will 1 node with 1 core each.

#SBATCH -n 1

#SBATCH -c 16

# Request memory

#SBATCH --mem=64G

# #SBATCH --mem-per-cpu=32

# Mail when the job begins, ends, fails, requeues

#SBATCH --mail-type=ALL

# Where to send email alerts

#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file that will be jet_ml_synthesis_pointnet_model_kfolded_all_datasets_output_<jobid>.out

#SBATCH -o jet_ml_synthesis_pointnet_model_kfolded_all_datasets_output_%j.out

# Create an error file that will be jet_ml_synthesis_pointnet_model_kfolded_all_datasets_error_<jobid>.out

#SBATCH -e jet_ml_synthesis_pointnet_model_kfolded_all_datasets_error_%j.err

# Set maximum time limit

#SBATCH -t 24:0:0

# List assigned GPU:

echo Assigned GPU: $CUDA_VISIBLE_DEVICES

# Check state of GPU:

nvidia-smi

#Converting jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to python jet_ml_synthesis_pointnet_model_kfolded_all_datasets.ipynb

echo "Setting up python version and conda shell"
ml python/3.7

source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh


#Activating conda environment
echo "Activating conda environment"
# conda activate tensorflow_env
conda activate tensorflow-gpu-v2.8

#Running simulation
echo "Running simulation"
python -u jet_ml_synthesis_pointnet_model_kfolded_all_datasets.py