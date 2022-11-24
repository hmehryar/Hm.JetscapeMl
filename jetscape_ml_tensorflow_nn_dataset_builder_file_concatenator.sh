#Author: H. Mehryar
#email: hmehryar@wayne.edu

#Converting jupyter notebook to python script
echo "Converting notebook to script"
jupyter nbconvert --to python jetscape_ml_tensorflow_nn_dataset_builder_file_concatenator.ipynb

python jetscape_ml_tensorflow_nn_dataset_builder_file_concatenator.py