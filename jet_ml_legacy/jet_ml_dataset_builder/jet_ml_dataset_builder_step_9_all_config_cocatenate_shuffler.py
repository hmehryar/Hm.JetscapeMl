#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

print('Loading/Installing Package => Begin\n\n')
# Commonly used modules
import numpy as np
import os
from os import path, makedirs
import time
from time import time
import subprocess


def install(package):
  print("Installing "+package) 
  subprocess.check_call([sys.executable,"-m" ,"pip", "install", package])
  print("Installed "+package+"\n") 



# Images, plots, display, and visualization
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

#reading/writing into files
# !pip3 install pickle5
install("pickle5")
import pickle5 as pickle

print('\n########################################################################')
print('Checking the running platforms\n')

from jet_ml_dataset_builder_utilities import set_directory_paths
# Call the function and retrieve the dataset_directory_path and simulation_directory_path
dataset_directory_path, simulation_directory_path = set_directory_paths()

# Access the dataset_directory_path and simulation_directory_path
print("Dataset Directory Path:", dataset_directory_path)
print("Simulation Directory Path:", simulation_directory_path)
print('########################################################################\n')


print('\nLoading/Installing Package => End\n\n')


# In[ ]:


from jet_ml_dataset_builder_utilities import  parse_parameters

# Call the function and retrieve the tokenized parameters
tokenized_arguments, tokenized_values = parse_parameters()

# Access the tokenized arguments and values
print("Tokenized Arguments:")
for argument in tokenized_arguments:
    print(argument)

print("\nTokenized Values:")
for argument, value in tokenized_values.items():
    print(f"{argument}: {value}")

y_class_label_items=['MMAT','MLBT']


# In[ ]:


from jet_ml_dataset_builder_utilities import load_dataset
from jet_ml_dataset_builder_utilities import merge_datasets

alpha_s_items=[0.2 ,0.3 ,0.4]
q0_items=[1.5 ,2.0 ,2.5]
class_labels = '_'.join(y_class_label_items)

def load_merge_add_configs():
    print ("Test: Loading dataset all config 1 to 9 datasets")


    # dataset_file_name='config_02_alpha_0.3_q0_1.5_MMAT_MLBT_size_1200000_shuffled.pkl'
    print("Loading load_datasets")
    dataset=None
    current_dataset=None
    # if dataset is None:
    configuration_number= 1
    
    # total_size_items=[1200000,1200000,1200000,1200000,1199991,1200000,1200000,1200000,1200000]
    total_size=1200000
    for q0 in q0_items:
        for alpha_s in alpha_s_items:
            print("------------------------")
            print("Loading configuration ",configuration_number)
            
            dataset_file_name = f"config_0{configuration_number}_alpha_{alpha_s}_q0_{q0}_{class_labels}_size_{total_size}_shuffled.pkl"
            dataset_file_name=simulation_directory_path+dataset_file_name
            print ("filename:",dataset_file_name)
            current_dataset = load_dataset(dataset_file_name,is_array=True)
            ((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))=current_dataset
            print("post load")
            print("dataset.x_train:",type(dataset_x_train), dataset_x_train.size, dataset_x_train.shape)
            print("dataset.x_test:",type(dataset_x_test), dataset_x_test.size, dataset_x_test.shape)
            print("dataset.y_train:",type(dataset_y_train), dataset_y_train.size,dataset_y_train.shape)
            print("dataset.y_test:",type(dataset_y_test), dataset_y_test.size, dataset_y_test.shape)

            print("Mergining configuration ",configuration_number)
            dataset = merge_datasets(dataset, current_dataset)

            print("------------------------")

            configuration_number=configuration_number+1

# load_merge_add_configs()


# In[ ]:


print("Building required params for the loading/saving the dataset file")
class_labels_str = '_'.join(y_class_label_items)
alpha_s_items_str='_'.join(map(str, alpha_s_items))
q0_items_str='_'.join(map(str, q0_items))
total_size=9*1200000
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}.pkl"
dataset_file_name=simulation_directory_path+dataset_file_name


# In[ ]:


from jet_ml_dataset_builder_utilities import load_dataset
print("Loading all config dataset")
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}-swapped.pkl"
dataset_file_name=simulation_directory_path+dataset_file_name
dataset=load_dataset (dataset_file_name)
((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))=dataset
print("dataset y_train values:\n", dataset_y_train[1:100])
print("dataset y_test values:\n", dataset_y_test[1:10])



# In[ ]:


from jet_ml_dataset_builder_utilities import save_dataset
# print ("Storing the dataset before shuffle")
# save_dataset(dataset_file_name,dataset)


# In[ ]:


from jet_ml_dataset_builder_utilities import load_dataset
def swap_y_train_with_x_test():
    print("Loading the dataset to check the structure before shuffle")
    dataset=load_dataset (dataset_file_name)
    ((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))=dataset
    temp=None
    temp=dataset_y_train
    dataset_y_train=dataset_x_test
    dataset_x_test=temp
    print("post swap")
    print("dataset.x_train:",type(dataset_x_train), dataset_x_train.size, dataset_x_train.shape)
    print("dataset.x_test:",type(dataset_x_test), dataset_x_test.size, dataset_x_test.shape)
    print("dataset.y_train:",type(dataset_y_train), dataset_y_train.size,dataset_y_train.shape)
    print("dataset.y_test:",type(dataset_y_test), dataset_y_test.size, dataset_y_test.shape)

    print("saving post swap dataset")
    dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}-swapped.pkl"
    dataset_file_name=simulation_directory_path+dataset_file_name
    dataset=((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))
    save_dataset(dataset_file_name,dataset)
# swap_y_train_with_x_test()


# In[ ]:


from jet_ml_dataset_builder_utilities import shuffle_dataset
shuffled_dataset = shuffle_dataset(dataset)


# In[ ]:


from jet_ml_dataset_builder_utilities import save_dataset
print("Saving Shuffled dataset")
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled.pkl"
dataset_file_name=simulation_directory_path+dataset_file_name
save_dataset(dataset_file_name,shuffled_dataset)

