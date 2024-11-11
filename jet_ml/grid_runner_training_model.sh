#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Check state of GPU:
nvidia-smi


# Get the parent dir

ROOT_PATH="jet_ml/"
# Define the file name as a variable
NOTEBOOK_PATH="classifiers/alpha_s/"
# NOTEBOOK_PATH="notebooks/"
FILE_NAME="alpha_s_transfer_learning_resnet50"
# FILE_NAME="building_alpha_s_dataset_with_constant_q0"
SERVER_NAME="$1"
# SERVER_NAME="wsu_grid_a100_cpu_8_mem_150gb"

# SERVER_NAME="wsu_grid_v100_cpu_24_mem_256gb"
# DATASET_SIZE="q0_2.5_1800k_batch_size_128"
# DATASET_SIZE="q0_2.5_250_batch_size_128"
DATASET_SIZE="100k_batch_size_128_no-lr"
# DATASET_SIZE="1k"
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
# PYTHON_SCRIPT="${RUNNER_SCRIPTS_PATH}${FILE_NAME}.py"
PYTHON_SCRIPT="${RUNNER_SCRIPTS_PATH}${FILE_NAME}_${DATASET_SIZE}_${SERVER_NAME}.py"
PYTHON_OUTPUT="${RUNNER_SCRIPTS_PATH}${FILE_NAME}_${DATASET_SIZE}_${SERVER_NAME}.output"

# PYTHON_SCRIPT="${FILE_NAME}.py"
echo "Running the following python script"
echo $PYTHON_SCRIPT
echo "##########################################"



# Converting Jupyter notebook to python script
# cd /wsu/home/gy/gy40/gy4065/hm_jetscapeml_source/jet_ml/classifiers/alpha_s
echo "Converting notebook to script"
# 
jupyter nbconvert --to python ${NOTEBOOK} --output ../../../${PYTHON_SCRIPT}
# jupyter nbconvert --to python ${NOTEBOOK} --output ../../${PYTHON_SCRIPT}

# Setting up python version and conda shell
# echo "Setting up python version and conda shell and environment on Grid"
ml python/3.7
source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh
conda init
conda activate tensorflow-gpu-v2.8

# echo "Setting up python version and conda shell and environment on HmSrv"
# conda init
# conda activate tensorflow


# Running simulation
echo "Running simulation"
python -u ../${PYTHON_SCRIPT} | tee ../${PYTHON_OUTPUT}