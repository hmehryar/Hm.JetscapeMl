#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name
##SBATCH --job-name=alphas-pointnet-250
##SBATCH --job-name=alphas-pointnet-100k
##SBATCH --job-name=alphas-pointnet-1000k

##SBATCH --job-name=alphas-resnet-250
#SBATCH --job-name=alphas-resnet-100k
##SBATCH --job-name=alphas-resnet-1000k

# Submit to the GPU QoS
##SBATCH -q primary
#SBATCH -q gpu

# Request the GPU type
#SBATCH --gres=gpu:2

# Request v100 gpu
# SBATCH --constraint=v100

# Total number of cores, in this example it will 1 node with 1 core each.
#SBATCH -n 2
#SBATCH -c 12

##SBATCH -N 1

# Request memory
#SBATCH --mem=256G


# Mail when the job begins, ends, fails, requeues
#SBATCH --mail-type=ALL

# Where to send email alerts
#SBATCH --mail-user=gy4065@wayne.edu

# Set maximum time limit
#SBATCH -t 100:0:0



# Create an output file
##SBATCH -o ../runner_scripts/alpha_s_pointnet_250_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_output_%j.out
##SBATCH -o ../runner_scripts/alpha_s_pointnet_100k_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_output_%j.out
##SBATCH -o ../runner_scripts/alpha_s_pointnet_1000K_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_output_%j.out

##SBATCH -o ../runner_scripts/alpha_s_resnet_250_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_output_%j.out
#SBATCH -o ../runner_scripts/alpha_s_resnet_100k_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_no_lr_output_%j.out
##SBATCH -o ../runner_scripts/alpha_s_resnet_1000K_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_output_%j.out

# Create an error file
##SBATCH -e ../runner_scripts/alpha_s_pointnet_250_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_error_%j.err
##SBATCH -e ../runner_scripts/alpha_s_pointnet_100k_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_error_%j.err
##SBATCH -o ../runner_scripts/alpha_s_pointnet_1000K_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_output_%j.out

##SBATCH -e ../runner_scripts/alpha_s_resnet_250_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_error_%j.err
#SBATCH -e ../runner_scripts/alpha_s_resnet_100k_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_no_lr_error_%j.err
##SBATCH -o ../runner_scripts/alpha_s_resnet_1000K_batch_size_128_wsu_grid_v100_cpu_24_mem_256gb_output_%j.out
$SERVER_NAME="wsu_grid_v100_cpu_24_mem_256gb"

./grid_runner_training_model.sh "$SERVER_NAME"