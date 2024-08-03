#!/bin/sh
#This file should be place in the parent directory that all other jetscape instances are, in this case Source directory

# Author: H. Mehryar
# email: hmehryar@wayne.edu

# Job name

#SBATCH --job-name s-matter-hadron-mover

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

# Create an output file that will be config-09-s-matter-hadron-mover-matter-0x-output-<jobid>.out

#SBATCH -o config-09-s-matter-hadron-mover-matter-0x-output-%j.out

# Create an error file that will be config-09-s-matter-hadron-mover-matter-0x-error-<jobid>.out

#SBATCH -e config-09-s-matter-hadron-mover-matter-0x-error-%j.err

# Set maximum time limit

#SBATCH -t 150:0:0

#User must assign the correct CONFIG_NUMBER after jetscape simulation is done
CONFIG_NUMBER=9

# MATTER, MATTERLBT
ELOSS_TYPE_UPPERCASE="MATTER"

# matter, matterlbt
ELOSS_TYPE_LOWERCASE="matter"


# MAJOR_PART_NUM=1
MINOR_PART_NUM=7



# echo "scheduling $ELOSS_TYPE_UPPERCASE-60K"
# for PART_NUM in 1 2 3 4 5 6
# do
#     echo "Part $PART_NUM: scheduling"
#     # cd JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-0$PART_NUM/JETSCAPE-COMP/build/
#     sbatch singularity-runner-$ELOSS_TYPE_LOWERCASE-0$PART_NUM.sh
#     echo "Part $PART_NUM: scheduled!"
# done
# 1 2 3 4 5 6
for MAJOR_PART_NUM in 4 5 6
do
    echo "rename/Moving $ELOSS_TYPE_UPPERCASE config-$CONFIG_NUMBER part$MAJOR_PART_NUM$MINOR_PART_NUM hadron file "

    echo "Config $CONFIG_NUMBER Part $MAJOR_PART_NUM$MINOR_PART_NUM: Renaming"
    mv JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-0$MAJOR_PART_NUM/JETSCAPE-COMP/build/finalStateHadrons.dat JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-0$MAJOR_PART_NUM/JETSCAPE-COMP/build/config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-$MAJOR_PART_NUM$MINOR_PART_NUM.dat
    echo "Config $CONFIG_NUMBER Part $MAJOR_PART_NUM$MINOR_PART_NUM: Done!"
    # echo "Config $CONFIG_NUMBER Part $MAJOR_PART_NUM$MINOR_PART_NUM: Renaming"
    # mv JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-$MAJOR_PART_NUM$MINOR_PART_NUM/JETSCAPE-COMP/build/finalStateHadrons.dat JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-$MAJOR_PART_NUM$MINOR_PART_NUM/JETSCAPE-COMP/build/config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-$MAJOR_PART_NUM$MINOR_PART_NUM.dat
    # echo "Config $CONFIG_NUMBER Part $MAJOR_PART_NUM$MINOR_PART_NUM: Done!"


    echo "Config $CONFIG_NUMBER Part $MAJOR_PART_NUM$MINOR_PART_NUM: Moving"
    mv JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-0$MAJOR_PART_NUM/JETSCAPE-COMP/build/config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-$MAJOR_PART_NUM$MINOR_PART_NUM.dat config-0$CONFIG_NUMBER-final-state-hadrons/
    echo "Config $CONFIG_NUMBER Part $MAJOR_PART_NUM$MINOR_PART_NUM: Done!"
done


# echo "Config $CONFIG_NUMBER Part $MAJOR_PART_NUM$MINOR_PART_NUM: Moving"
# mv JETSCAPE-ML-HAYDAR-$ELOSS_TYPE_UPPERCASE-$MAJOR_PART_NUM$MINOR_PART_NUM/JETSCAPE-COMP/build/config-0$CONFIG_NUMBER-final-state-hadrons-$ELOSS_TYPE_LOWERCASE-$MAJOR_PART_NUM$MINOR_PART_NUM.dat config-0$CONFIG_NUMBER-final-state-hadrons/
# echo "Config $CONFIG_NUMBER Part $MAJOR_PART_NUM$MINOR_PART_NUM: Done!"


# for PART_NUM in 1 2 3 4 5 6
# do

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


