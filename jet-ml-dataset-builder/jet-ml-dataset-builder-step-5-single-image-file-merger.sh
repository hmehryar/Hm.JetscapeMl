#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name e-img-chunk-merger-cnf-09-mat

#SBATCH -q primary
# # Submit to the GPU QoS

# #SBATCH -q gpu

# # Request the GPU type

# #SBATCH --gres=gpu:tesla

# Total number of cores, in this example it will 1 node with 1 core each.

#SBATCH -n 32

# Request memory

#SBATCH --mem=128G

# Mail when the job begins, ends, fails, requeues

#SBATCH --mail-type=ALL

# Where to send email alerts

#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file that will be event-to-image-chunk-merger-config-09-mat-output-<jobid>.out

#SBATCH -o event-to-image-chunk-merger-config-09-mat-output-%j.out

# Create an error file that will be event-to-image-chunk-merger-config-09-mat-error-<jobid>.out

#SBATCH -e event-to-image-chunk-merger-config-09-mat-error-%j.err

# Set maximum time limit

#SBATCH -t 100:0:0

#Converting jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to python jet-ml-dataset-builder-step-5-single-image-file-merger.ipynb

echo "Setting up python version and conda shell"
ml python/3.7

source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh


#Activating conda environment
echo "Activating conda environment"
# conda activate tensorflow_env
conda activate tensorflow_gpuenv_v2

#Execution
echo "Running image file merger"

#User must assign the correct CONFIG_NUMBER after jetscape simulation is done
CONFIG_NUMBER=9


# MLBT, MMAT
ELOSS_TYPE_UPPERCASE="MMAT"

# matter, matterlbt
ELOSS_TYPE_LOWERCASE="matter"

echo "Running chunck image merger for $CONFIG_NUMBER"

# python jetscape-ml-tensorflow-nn-dataset-builder-single-file-event-splitter.py -i finalStateHadrons-Matter-1k.dat -d 1000 -y MVAC -o jetscape-ml-benchmark-dataset-600k-matter.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER
# python jetscape-ml-tensorflow-nn-dataset-builder-single-file-event-to-image-convertor.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER
python jet-ml-dataset-builder-step-5-single-image-file-merger.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER


# # MVAC, MLBT, MMAT
# ELOSS_TYPE_UPPERCASE="MLBT"

# # matter, matterlbt
# ELOSS_TYPE_LOWERCASE="matterlbt"
# python jetscape-ml-tensorflow-nn-dataset-builder-single-image-file-merger.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER