#!/bin/bash

# Job name

#SBATCH --job-name Tensorflow

# Submit to the GPU QoS

#SBATCH -q gpu

# Request one node

#SBATCH -N 1

# Total number of cores, in this example it will 1 node with 1 core each.

#SBATCH -n 32

# Request memory

#SBATCH --mem=128G

# Mail when the job begins, ends, fails, requeues

#SBATCH --mail-type=ALL

# Where to send email alerts

#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file that will be hm_jetscape_ml_cnn_output_<jobid>.out

#SBATCH -o hm_jetscape_ml_cnn_output_%j.out

# Create an error file that will be hm_jetscape_ml_cnn_error_<jobid>.out

#SBATCH -e hm_jetscape_ml_cnn_errors_%j.err

# Set maximum time limit

#SBATCH -t 36:0:0


ml python/3.7

source /wsu/e17/pre-compiled/python/3.7/etc/profile.d/conda.sh

conda activate tensorflow_env

python jetscape-ml-tensorflow-cnn.py