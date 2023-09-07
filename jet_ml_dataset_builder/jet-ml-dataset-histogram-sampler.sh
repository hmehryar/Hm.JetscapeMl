#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name e-splitter-config-09-mat

# Submit to the GPU QoS

#SBATCH -q gpu

# Request the GPU type

#SBATCH --gres=gpu:tesla

# Total number of cores, in this example it will 1 node with 1 core each.

#SBATCH -n 4

# Request memory

#SBATCH --mem=16G

# Mail when the job begins, ends, fails, requeues

#SBATCH --mail-type=ALL

# Where to send email alerts

#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file that will be event-splitter-config-09-mat-output-<jobid>.out

#SBATCH -o event-splitter-config-09-mat-output-%j.out

# Create an error file that will be event-splitter-config-09-mat-error-<jobid>.out

#SBATCH -e event-splitter-config-09-mat-error-%j.err

# Set maximum time limit

#SBATCH -t 100:0:0

#Converting jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to python jet-ml-dataset-histogram-sampler.ipynb

echo "Setting up python version and conda shell"
ml python/3.7

# source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh


#Activating conda environment
echo "Activating conda environment"
conda activate tensorflow_gpuenv_v2


#Execution
echo "Running dataset histogram sampler"

# MMAT, MLBT
# matter, matterlbt

#User must assign the correct CONFIG_NUMBER after jetscape simulation is done
#####################################
#1
CONFIG_NUMBER=1
ELOSS_TYPE_UPPERCASE="MMAT"
ELOSS_TYPE_LOWERCASE="matter"
Q_0=1
ALPHA_S=0.2
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#2
CONFIG_NUMBER=1
ELOSS_TYPE_UPPERCASE="MLBT"
ELOSS_TYPE_LOWERCASE="matterlbt"
Q_0=1.5
ALPHA_S=0.2
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#3
CONFIG_NUMBER=2
ELOSS_TYPE_UPPERCASE="MMAT"
ELOSS_TYPE_LOWERCASE="matter"
Q_0=1
ALPHA_S=0.3
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#4
CONFIG_NUMBER=2
ELOSS_TYPE_UPPERCASE="MLBT"
ELOSS_TYPE_LOWERCASE="matterlbt"
Q_0=1.5
ALPHA_S=0.3
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#5
CONFIG_NUMBER=3
ELOSS_TYPE_UPPERCASE="MMAT"
ELOSS_TYPE_LOWERCASE="matter"
Q_0=1
ALPHA_S=0.4
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#6
CONFIG_NUMBER=3
ELOSS_TYPE_UPPERCASE="MLBT"
ELOSS_TYPE_LOWERCASE="matterlbt"
Q_0=1.5
ALPHA_S=0.4
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#7
CONFIG_NUMBER=4
ELOSS_TYPE_UPPERCASE="MMAT"
ELOSS_TYPE_LOWERCASE="matter"
Q_0=1
ALPHA_S=0.2
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#8
CONFIG_NUMBER=4
ELOSS_TYPE_UPPERCASE="MLBT"
ELOSS_TYPE_LOWERCASE="matterlbt"
Q_0=2
ALPHA_S=0.2
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#9
CONFIG_NUMBER=5
ELOSS_TYPE_UPPERCASE="MMAT"
ELOSS_TYPE_LOWERCASE="matter"
Q_0=1
ALPHA_S=0.3
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#10
CONFIG_NUMBER=5
ELOSS_TYPE_UPPERCASE="MLBT"
ELOSS_TYPE_LOWERCASE="matterlbt"
Q_0=2
ALPHA_S=0.3
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#11
CONFIG_NUMBER=6
ELOSS_TYPE_UPPERCASE="MMAT"
ELOSS_TYPE_LOWERCASE="matter"
Q_0=1
ALPHA_S=0.4
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#12
CONFIG_NUMBER=6
ELOSS_TYPE_UPPERCASE="MLBT"
ELOSS_TYPE_LOWERCASE="matterlbt"
Q_0=2
ALPHA_S=0.4
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#13
CONFIG_NUMBER=7
ELOSS_TYPE_UPPERCASE="MMAT"
ELOSS_TYPE_LOWERCASE="matter"
Q_0=1
ALPHA_S=0.2
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#14
CONFIG_NUMBER=7
ELOSS_TYPE_UPPERCASE="MLBT"
ELOSS_TYPE_LOWERCASE="matterlbt"
Q_0=2.5
ALPHA_S=0.2
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#15
CONFIG_NUMBER=8
ELOSS_TYPE_UPPERCASE="MMAT"
ELOSS_TYPE_LOWERCASE="matter"
Q_0=1
ALPHA_S=0.3
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#16
CONFIG_NUMBER=8
ELOSS_TYPE_UPPERCASE="MLBT"
ELOSS_TYPE_LOWERCASE="matterlbt"
Q_0=2.5
ALPHA_S=0.3
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#17
CONFIG_NUMBER=9
ELOSS_TYPE_UPPERCASE="MMAT"
ELOSS_TYPE_LOWERCASE="matter"
Q_0=1
ALPHA_S=0.4
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################
#####################################
#18
CONFIG_NUMBER=9
ELOSS_TYPE_UPPERCASE="MLBT"
ELOSS_TYPE_LOWERCASE="matterlbt"
Q_0=2.5
ALPHA_S=0.4
echo "Running histogram sample on Config $CONFIG_NUMBER $ELOSS_TYPE_LOWERCASE"
python jet-ml-dataset-histogram-sampler.py -i config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat -d 600000 -y $ELOSS_TYPE_UPPERCASE -o jetscape-ml-benchmark-dataset-600k-$ELOSS_TYPE_LOWERCASE.pkl -n 40 -c ~/Projects/110_JetscapeMl/Source/config-0$CONFIG_NUMBER-final-state-hadrons/ -p $CONFIG_NUMBER -q $Q_0 -a $ALPHA_S
#####################################