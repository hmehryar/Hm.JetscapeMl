#!/bin/sh
#This file should be place in the parent directory that all other jetscape instances are, in this case Source directory

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name s-event-merger-config-09-mat

# Submit to the GPU QoS
#SBATCH -q primary

# #SBATCH -q gpu

# # Request the GPU type

# #SBATCH --gres=gpu:tesla

# Total number of cores, in this example it will 1 node with 1 core each.

#SBATCH -n 4

# Request memory

#SBATCH --mem=16G

# Mail when the job begins, ends, fails, requeues

#SBATCH --mail-type=ALL

# Where to send email alerts

#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file that will be config-09-s-events-merger-600k-matter-output-<jobid>.out

#SBATCH -o config-09-s-events-merger-600k-matter-output--%j.out

# Create an error file that will be config-09-s-events-merger-600k-matter-error-<jobid>.out

#SBATCH -e config-09-s-events-merger-600k-matter-error-%j.err

# Set maximum time limit

#SBATCH -t 150:0:0

#User must assign the correct CONFIG_NUMBER after jetscape simulation is done
CONFIG_NUMBER=9

# MATTER, MATTERLBT
ELOSS_TYPE_UPPERCASE="MATTER"

# matter, matterlbt
ELOSS_TYPE_LOWERCASE="matter"

echo "Merging $CONFIG_NUMBER: $ELOSS_TYPE_UPPERCASE-600K"

cd config-0$CONFIG_NUMBER-final-state-hadrons/
pwd

touch  config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat


#This piece of code is for the time when the events are splitted in 10k and 20k
# 0 -> when there are 100K chunks
# 1 2 3 4 5 6 -> where there are 15K or 10K chunks
for PART_NUM in 1 2 3 4 5 6
do
    # 0
    for SUB_PART_NUM in 0 1 2 3 4 5 6  7
    do
        echo "Part $PART_NUM$SUB_PART_NUM: Processing JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE"
        cat config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-$PART_NUM$SUB_PART_NUM.dat >> config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat
        echo "Part $PART_NUM$SUB_PART_NUM: Done!"
    done
    
done


