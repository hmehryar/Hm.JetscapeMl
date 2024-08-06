#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Define the file name as a variable
DIR_PATH="jet_ml\classifiers\alpha_s"
FILE_NAME="alpha_s_test_net"
JOB_NAME="${FILE_NAME}-100k"
OUTPUT_FILE="${FILE_NAME}_output_%j.out"
ERROR_FILE="${FILE_NAME}_error_%j.err"
PYTHON_SCRIPT="${FILE_NAME}_gpu.py"
NOTEBOOK="${FILE_NAME}.ipynb"

# Job name
#SBATCH --job-name=${JOB_NAME}

# Submit to the GPU QoS
##SBATCH -q primary
#SBATCH -q gpu

# Request the GPU type
##SBATCH --gres=gpu:tesla
# #SBATCH --gres=gpu:geforc
#SBATCH --gres=gpu:2

# Request v100 gpu
#SBATCH --constraint=v100

# Total number of cores, in this example it will 1 node with 1 core each.
#SBATCH -n 32
#SBATCH -c 1

# Request memory
#SBATCH --mem=120G
# #SBATCH --mem-per-cpu=32

# Mail when the job begins, ends, fails, requeues
#SBATCH --mail-type=ALL

# Where to send email alerts
#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file
#SBATCH -o ${OUTPUT_FILE}

# Create an error file
#SBATCH -e ${ERROR_FILE}

# Set maximum time limit
#SBATCH -t 500:0:0

# List assigned GPU:
echo Assigned GPU: $CUDA_VISIBLE_DEVICES

# Check state of GPU:
nvidia-smi

# Converting Jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to python ${NOTEBOOK} --output ${PYTHON_SCRIPT}

# Setting up python version and conda shell
echo "Setting up python version and conda shell"
ml python/3.7
source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh

# Activating conda environment
echo "Activating conda environment"
# conda activate tensorflow_env
conda activate tensorflow-gpu-v2.8

# Running simulation
echo "Running simulation"
python -u ${PYTHON_SCRIPT} | tee ${FILE_NAME}_gpu.output
