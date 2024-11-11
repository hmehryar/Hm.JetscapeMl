#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name
##SBATCH --job-name=alphas-pointnet-250
##SBATCH --job-name=alphas-pointnet-100k
##SBATCH --job-name=alphas-pointnet-1000k

# Submit to the GPU QoS
##SBATCH -q primary
##SBATCH -q gpu

#SBATCH -q express
#SBATCH -p ecscp

# Request the GPU type
##SBATCH --gres=gpu:tesla
##SBATCH --gres=gpu:geforc
##SBATCH --gres=gpu:2
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1

# Request v100 gpu
## SBATCH --constraint=v100

# Total number of cores, in this example it will 1 node with 1 core each.
#SBATCH -n 1
#SBATCH -c 8


##SBATCH -N 1

# Request memory
##SBATCH --mem=256G
##SBATCH --mem-per-cpu=32
#SBATCH --mem=150G

# Mail when the job begins, ends, fails, requeues
#SBATCH --mail-type=ALL

# Where to send email alerts
#SBATCH --mail-user=gy4065@wayne.edu

# Set maximum time limit
#SBATCH -t 100:0:0



# Create an output file
##SBATCH -o ../runner_scripts/alpha_s_pointnet_250_batch_size_128_wsu_grid_a100_cpu_8_mem_150gb_output_%j.out
#SBATCH -o ../runner_scripts/alpha_s_pointnet_100k_batch_size_128_wsu_grid_a100_cpu_8_mem_150gb_output_%j.out
## SBATCH -o ../runner_scripts/alpha_s_pointnet_1000K_batch_size_128_wsu_grid_a100_cpu_8_mem_150gb_output_%j.out

# Create an error file
##SBATCH -e ../runner_scripts/alpha_s_pointnet_250_batch_size_128_wsu_grid_a100_cpu_8_mem_150gb_error_%j.err
#SBATCH -e ../runner_scripts/alpha_s_pointnet_100k_batch_size_128_wsu_grid_a100_cpu_8_mem_150gb_error_%j.err
##SBATCH -o ../runner_scripts/alpha_s_pointnet_1000K_batch_size_128_wsu_grid_a100_cpu_8_mem_150gb_output_%j.out
$SERVER_NAME="wsu_grid_a100_cpu_8_mem_150gb"

./grid_runner_training_model.sh "$SERVER_NAME"