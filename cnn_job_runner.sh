#Author: H. Mehryar
#email: hmehryar@wayne.edu

#Converting jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to script jetscape-ml-tensorflow-cnn.ipynb

echo "Setting up python version and conda shell"
ml python/3.7

source /wsu/el7/pre-compiled/python/3.7/etc/profile.d/conda.sh


#Activating conda environment
echo "Activating conda environment"
# conda activate tensorflow_env
conda activate tensorflow_gpuenv_v2

#Running simulation
echo "Running simulation"
python jetscape-ml-tensorflow-cnn.py