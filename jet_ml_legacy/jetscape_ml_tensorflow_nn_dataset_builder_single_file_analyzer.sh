#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name jet_ml_dataset_builder_single_file_analyzer_config_01_matt

# Submit to the GPU QoS

#SBATCH -q primary
# #SBATCH -q gpu

# # Request the GPU type

# #SBATCH --gres=gpu:tesla

# Total number of cores, in this example it will 1 node with 1 core each.

#SBATCH -n 32

# Request memory

#SBATCH --mem=256G

# Mail when the job begins, ends, fails, requeues

#SBATCH --mail-type=ALL

# Where to send email alerts

#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file that will be jet_ml_dataset_builder_single_file_analyzer_output-<jobid>.out

#SBATCH -o jet_ml_dataset_builder_single_file_analyzer_output-%j.out

# Create an error file that will be jet_ml_dataset_builder_single_file_analyzer_error-<jobid>.out

#SBATCH -e jet_ml_dataset_builder_single_file_analyzer_error-%j.err

# Set maximum time limit

#SBATCH -t 150:0:0

#Converting jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to python jetscape_ml_tensorflow_nn_dataset_builder_single_file_analyzer.ipynb

echo "Setting up python version and conda shell"
ml python/3.7

source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh


#Activating conda environment
echo "Activating conda environment"
# conda activate tensorflow_env
conda activate tensorflow_gpuenv_v2


#User must assign the correct CONFIG_NUMBER after jetscape simulation is done
CONFIG_NUMBER=1


# MVAC, MLBT, MMAT
ELOSS_TYPE_UPPERCASE="MMAT"

# matter, matterlbt
ELOSS_TYPE_LOWERCASE="matter"

echo "Running single file analyzer for config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"


#Running simulation
echo "Running simulation"
echo "Running simulation For MATTER"
#python jetscape_ml_tensorflow_nn_dataset_builder_single_file_analyzer.py -i finalStateHadrons-Matter-1k.dat -d 1000 -y MVAC -o jetscape-ml-benchmark-dataset-1k-matter.pkl
python jetscape_ml_tensorflow_nn_dataset_builder_single_file_analyzer.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER