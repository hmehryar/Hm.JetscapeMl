#!/bin/sh
#This file should be place in the parent directory that all other jetscape instances are, in this case Source directory

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name s-batch-runner-matter-09

# Submit to the GPU QoS

#SBATCH -q primary

# #SBATCH -q gpu

# # Request the GPU type

# #SBATCH --gres=gpu:tesla

# Total number of cores, in this example it will 1 node with 1 core each.

#SBATCH -n 1

# Request memory

#SBATCH --mem=1G

# Mail when the job begins, ends, fails, requeues

#SBATCH --mail-type=ALL

# Where to send email alerts

#SBATCH --mail-user=gy4065@wayne.edu

# Create an output file that will be config-09-s-batch-runner-matter-output-<jobid>.out

#SBATCH -o config-09-s-batch-runner-matter-output-%j.out

# Create an error file that will be config-09-s-batch-runner-matter-error-<jobid>.out

#SBATCH -e config-09-s-batch-runner-matter-error-%j.err

# Set maximum time limit

#SBATCH -t 150:0:0

#User must assign the correct CONFIG_NUMBER after jetscape simulation is done
CONFIG_NUMBER=9

# MATTER, MATTERLBT
ELOSS_TYPE_UPPERCASE="MATTER"

# matter, matterlbt
ELOSS_TYPE_LOWERCASE="matter"

echo "scheduling $ELOSS_TYPE_UPPERCASE-600K"
# 1 
for PART_NUM in 3 4 5 6
do
    echo "Part $PART_NUM: scheduling"
    # cd JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-0$PART_NUM/JETSCAPE-COMP/build/
    sbatch singularity-runner-$ELOSS_TYPE_LOWERCASE-0$PART_NUM.sh
    echo "Part $PART_NUM: scheduled!"
done

# for PART_NUM in 1 2 3 4 5 6
# do
#     echo "Part $PART_NUM: Renaming"
#     mv JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-0$PART_NUM/JETSCAPE-COMP/build/finalStateHadrons.dat JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-0$PART_NUM/JETSCAPE-COMP/build/config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-0$PART_NUM.dat
#     echo "Part $PART_NUM: Done!"
# done


# for PART_NUM in 1 2 3 4 5 6 
# do
#     echo "Part $PART_NUM: Moving"
#     mv JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-0$PART_NUM/JETSCAPE-COMP/build/config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-0$PART_NUM.dat config-0$CONFIG_NUMBER-final-state-hadrons/
#     echo "Part $PART_NUM: Done!"
# done

# cd config-0$CONFIG_NUMBER-final-state-hadrons/
# pwd

# touch  config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat


# for PART_NUM in 1 2 3 4 5 6
# do
#     echo "Part $PART_NUM: Processing JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE"
#     cat config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-0$PART_NUM.dat >> config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-600k.dat
#     echo "Part $PART_NUM: Done!"
# done


