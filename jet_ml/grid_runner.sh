#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name
#SBATCH --job-name=a-s-point-q-2-1800k-b128-nolr

# Submit to the GPU QoS
#SBATCH -q gpu

##SBATCH -q express
##SBATCH -p ecscp

# Request the GPU type
#SBATCH --gres=gpu:2
##SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1

# Request v100 gpu
# SBATCH --constraint=v100

# Total number of cores, in this example it will 1 node with 1 core each.
#SBATCH -n 2
#SBATCH -c 12

##SBATCH -n 1
##SBATCH -c 8

##SBATCH -N 1

# Request memory
#SBATCH --mem=256G
##SBATCH --mem=150G

# Mail when the job begins, ends, fails, requeues
#SBATCH --mail-type=ALL

# Where to send email alerts
#SBATCH --mail-user=gy4065@wayne.edu

# Set maximum time limit
#SBATCH -t 500:0:0

# Set the output and error log based on the simulation name
# Create an output file
#SBATCH -o ../runner_scripts/alpha_s_pointnet_q0_2.0_1800k_batch_size_128_no_lr_wsu_grid_v100_cpu_24_mem_256gb_output_%j.out

# Create an error file
#SBATCH -e ../runner_scripts/alpha_s_pointnet_q0_2.0_1800k_batch_size_128_no_lr_wsu_grid_v100_cpu_24_mem_256gb_error_%j.err


# Check state of GPU:
nvidia-smi


# Get the parent dir

ROOT_PATH="jet_ml/"
# Define the file name as a variable
NOTEBOOK_PATH="classifiers/alpha_s/"
# NOTEBOOK_PATH="notebooks/"

FILE_NAME="alpha_s_pointnet"
# FILE_NAME="building_alpha_s_dataset_with_constant_q0"

# SERVER_NAME="wsu_grid_a100_cpu_8_mem_150gb"
SERVER_NAME="wsu_grid_v100_cpu_24_mem_256gb"

DATASET_SIZE="q0_2.0_1800k_batch_size_128_no_rl"
# DATASET_SIZE="q0_2.5_250_batch_size_128"
# DATASET_SIZE="1000k_batch_size_128_no_rl"

JOB_NAME="${FILE_NAME}_${DATASET_SIZE}_${SERVER_NAME}"
OUTPUT_FILE="${JOB_NAME}_output_%j.out"
ERROR_FILE="${JOB_NAME}_error_%j.err"


NOTEBOOK="${NOTEBOOK_PATH}${FILE_NAME}.ipynb"
echo "Current directory"
pwd
echo "##########################################"
echo "Running the following notebook"
echo $NOTEBOOK
echo "##########################################"
RUNNER_SCRIPTS_PATH="runner_scripts/"
PYTHON_SCRIPT="${RUNNER_SCRIPTS_PATH}${FILE_NAME}_${DATASET_SIZE}_${SERVER_NAME}.py"
PYTHON_OUTPUT="${RUNNER_SCRIPTS_PATH}${FILE_NAME}_${DATASET_SIZE}_${SERVER_NAME}.output"

echo "Running the following python script"
echo $PYTHON_SCRIPT
echo "##########################################"



# Converting Jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to python ${NOTEBOOK} --output ../../../${PYTHON_SCRIPT}
# jupyter nbconvert --to python ${NOTEBOOK} --output ../../${PYTHON_SCRIPT}

# Setting up python version and conda shell
echo "Setting up python version and conda shell and environment on Grid"
ml python/3.7
source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh
conda init
conda activate tensorflow-gpu-v2.8

# echo "Setting up python version and conda shell and environment on HmSrv"
# conda init
# conda activate tensorflow

echo "Running simulation"
python -u ../${PYTHON_SCRIPT} | tee ../${PYTHON_OUTPUT}