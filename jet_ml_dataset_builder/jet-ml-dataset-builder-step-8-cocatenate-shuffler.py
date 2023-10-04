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


from jet_ml_dataset_builder_utilities import load_dataset_by_y_class_label
print ("Test: Loading dataset MATTER side and Printing Y side to make sure, 3 colums are")

print("Loading dataset_mvac")
dataset_mvac=None
if dataset_mvac is None:
    dataset_mvac=load_dataset_by_y_class_label(tokenized_values["configuration_number"],tokenized_values["data_size"],simulation_directory_path,y_class_label_items[0],tokenized_values["alpha_s"],q0=1)
    ((dataset_mvac_x_train,dataset_mvac_y_train),(dataset_mvac_x_test,dataset_mvac_y_test))=dataset_mvac

print("Loading dataset_mlbt")
dataset_mlbt=None
if dataset_mlbt is None:
    dataset_mlbt=load_dataset_by_y_class_label(tokenized_values["configuration_number"],tokenized_values["data_size"],simulation_directory_path,y_class_label_items[1],tokenized_values["alpha_s"],tokenized_values["q0"])
    ((dataset_mlbt_x_train,dataset_mlbt_y_train),(dataset_mlbt_x_test,dataset_mlbt_y_test))=dataset_mlbt


# In[ ]:


from jet_ml_dataset_builder_utilities import merge_and_shuffle_datasets
merged_dataset = merge_and_shuffle_datasets(dataset_mvac, dataset_mlbt)


# In[ ]:


from jet_ml_dataset_builder_utilities import store_merged_dataset
total_size = len(merged_dataset['x_train'])+len(merged_dataset['x_test'])
store_merged_dataset(merged_dataset,  tokenized_values["alpha_s"], tokenized_values["q0"], total_size, y_class_label_items, tokenized_values["configuration_number"],simulation_directory_path)

