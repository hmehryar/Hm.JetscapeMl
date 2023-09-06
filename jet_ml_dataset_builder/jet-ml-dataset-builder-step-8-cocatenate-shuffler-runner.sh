#!/bin/bash

# Author: H. Mehryar
# email: hmehryar@wayne.edu



let "CONFIG_NUMBER = 1"

# Loop over q0 values
for Q0 in 1.5 2.0 2.5; do
  # Loop over alpha_s values
  for ALPHA_S in 0.2 0.3 0.4; do
    # Create a separate job submission for each combination
    echo "#!/bin/bash" > jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "#SBATCH --job-name ds-con-shuf-cnf${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "#SBATCH -q primary" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "#SBATCH -n 4" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "#SBATCH --mem=100G" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "#SBATCH --mail-type=ALL" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "#SBATCH --mail-user=gy4065@wayne.edu" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "#SBATCH -o ds-con-shuf-config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}-output-%j.out" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "#SBATCH -e ds-con-shuf-config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}-error-%j.err" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "#SBATCH -t 100:0:0" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "echo "Converting notebook to script"" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "jupyter nbconvert --to python jet-ml-dataset-builder-step-8-cocatenate-shuffler.ipynb" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "echo \"Running dataset step-8-dataset-concat-shuffler for CONFIG_NUMBER $CONFIG_NUMBER, q0=$Q0, alpha_s=$ALPHA_S\"" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "python jet-ml-dataset-builder-step-8-cocatenate-shuffler.py -d 600000 -p $CONFIG_NUMBER -a $ALPHA_S -q $Q0" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    echo "" >> jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh

    # Submit the job
    sbatch jet-ml-dataset-builder-step-8-cocatenate-shuffler-job_script_config${CONFIG_NUMBER}_q0${Q0}_alpha${ALPHA_S}.sh
    let "CONFIG_NUMBER = CONFIG_NUMBER + 1"
  done
done

