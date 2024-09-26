#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Get the parent dir
ROOT_PATH="jet_ml/"
# Define the file name as a variable
NOTEBOOK_PATH="classifiers/alpha_s/"
FILE_NAME="alpha_s_transfer_learning_resnet50"
JOB_NAME="${FILE_NAME}-100k"
OUTPUT_FILE="${JOB_NAME}_output_%j.out"
ERROR_FILE="${JOB_NAME}_error_%j.err"
NOTEBOOK="${NOTEBOOK_PATH}${FILE_NAME}.ipynb"
echo "Current directory"
pwd
echo "##########################################"
echo "Running the following notebook"
echo $NOTEBOOK
echo "##########################################"
RUNNER_SCRIPTS_PATH="../../../runner_scripts/"
PYTHON_SCRIPT="${RUNNER_SCRIPTS_PATH}${FILE_NAME}.py"
# PYTHON_SCRIPT="${FILE_NAME}.py"
echo "Running the following python script"
echo $PYTHON_SCRIPT
echo "##########################################"


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
#echo Assigned GPU: $CUDA_VISIBLE_DEVICES

# Check state of GPU:
#nvidia-smi

# Converting Jupyter notebook to python script
# cd /wsu/home/gy/gy40/gy4065/hm_jetscapeml_source/jet_ml/classifiers/alpha_s
echo "Converting notebook to script"
jupyter nbconvert --to python ${NOTEBOOK} --output ${PYTHON_SCRIPT}

# Setting up python version and conda shell
#echo "Setting up python version and conda shell"
ml python/3.7
source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh

# Activating conda environment
echo "Activating conda environment"
# conda activate tensorflow_env
# conda init
# conda activate tensorflow
conda activate tensorflow-gpu-v2.8

# Running simulation
#echo "Running simulation"
python -u ${PYTHON_SCRIPT} | tee ${FILE_NAME}.output
