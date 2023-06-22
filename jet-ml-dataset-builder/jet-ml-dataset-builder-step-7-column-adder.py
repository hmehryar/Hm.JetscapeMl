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
import platform
running_os=platform.system()
print("OS: "+running_os)
print("OS version: "+platform.release())

try:
  from google.colab import drive
  COLAB = True
except:
  COLAB = False
print("running on Colab: "+str(COLAB))


print("Python version: "+platform.python_version())


dataset_directory_path=''
simulation_directory_path=''

if COLAB == True:
  drive.mount('/content/drive')
  dataset_directory_path='/content/drive/MyDrive/Projects/110_JetscapeMl/hm.jetscapeml.data/'
  simulation_directory_path=dataset_directory_path+'simulation_results/'
elif 'Linux' in running_os:
  dataset_directory_path='/wsu/home/gy/gy40/gy4065/hm.jetscapeml.data/'
  simulation_directory_path=dataset_directory_path+'simulation_results/'
else:
  dataset_directory_path= 'G:\\My Drive\\Projects\\110_JetscapeMl\\hm.jetscapeml.data\\'
  simulation_directory_path=dataset_directory_path+'simulation_results\\'
print('Dataset Directory Path: '+dataset_directory_path)



if not path.exists(simulation_directory_path):
    makedirs(simulation_directory_path)
print('Simulation Results Path: '+simulation_directory_path)
print('########################################################################\n')


print('\nLoading/Installing Package => End\n\n')


# In[ ]:


print('########################################################################\n')
print("Parsing parameters from command line and initializing the input parameters")
# Python program to demonstrate
# command line arguments
 
 
import getopt, sys
 
 
# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]
 
# Options
options = "hi:d:y:o:n:c:p:a:q:"
 
# Long options
long_options = ["Help", "Input_file_name_hadrons","Data_size","Y_class_label_items","output_dataset_file_name=", "number_of_partition","configuration_directory","configuration_number","alpha_s","q0"]
 
try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)
    print(arguments)
    print(values)
    # checking each argument
    for currentArgument, currentValue in arguments:
        print(currentArgument)
        if currentArgument in ("-h", "--Help"):
            print ("Displaying Help")   
        elif currentArgument in ("-i", "--Input_file_name_hadrons"):
            print ("Input_file_name_hadrons: ", currentValue)
            file_name_hadrons=currentValue
            print('simulated events final state hadron file: '+file_name_hadrons)
        elif currentArgument in ("-d", "--Data_size"):
            print ("Data_size: ", currentValue) 
            data_size=int(currentValue)
            print('data_size: {} '.format(data_size))
        elif currentArgument in ("-y", "--Y_class_label_items"):
            print ("Y_class_label_items: ", currentValue)
            y_class_label_items=[currentValue]     
            print("y_class_label_items")
            print(y_class_label_items)
        elif currentArgument in ("-o", "--output_dataset_file_name"):
            print ("output_dataset_file_name: ",currentValue)
            dataset_file_name=currentValue
            print("Dataset file name: "+dataset_file_name)
        elif currentArgument in ("-n", "--number_of_partition"):
            print ("number_of_partition: ",currentValue)
            number_of_partition=int(currentValue)
            print('Number of partition for splitting the events: {} '.format(number_of_partition))
        elif currentArgument in ("-c", "--configuration_directory"):
            print ("configuration_directory: ",currentValue)
            configuration_directory=currentValue
            print('Configuration directory: ',configuration_directory)
        elif currentArgument in ("-p", "--configuration_number"):
            print ("configuration_number: ",currentValue)
            configuration_number=int(currentValue)
            print('Configuration number to reference which dataset it is: {} '.format(configuration_number))
        elif currentArgument in ("-a", "--alpha_s"):
            print ("alpha_s: ",currentValue)
            alpha_s=float(currentValue)
            print('alpha_s float value: {} '.format(alpha_s))
        elif currentArgument in ("-q", "--q0"):
            print ("q0: ",currentValue)
            q0=float(currentValue)
            print('Q0 float value: {} '.format(q0))
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
print('########################################################################\n')


# In[ ]:


from jet_ml_dataset_builder_utilities import load_event_items_chunk
def load_x_set():
    print('\n########################################################################')
    start = time()
    print ("Loading X dataset")
    file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-img-chunk.pkl"
    file_name=simulation_directory_path+file_name
    event_items_image=load_event_items_chunk(file_name)
    
    print ("Loaded: ",file_name)
    print("Dataset type: ", type(event_items_image))
    print("Dataset length: ", len(event_items_image))
    print('\n########################################################################')
    
    elapsed = time() - start
    print('Loading x Dataset Elapsed %.3f seconds.' % elapsed)
    print('\n########################################################################')
    return (event_items_image)
# (x)=load_x_set()


# In[ ]:


from jet_ml_dataset_builder_utilities import load_dataset_by_y_class_label
y_class_label_items=['MMAT','MLBT']

print("Loading dataset_mvac")
dataset_mvac=None
if dataset_mvac is None:
    dataset_mvac=load_dataset_by_y_class_label(configuration_number,data_size,simulation_directory_path,y_class_label_items[0])
    ((dataset_mvac_x_train,dataset_mvac_y_train),(dataset_mvac_x_test,dataset_mvac_y_test))=dataset_mvac

print("Loading dataset_mlbt")
dataset_mlbt=None
if dataset_mlbt is None:
    dataset_mlbt=load_dataset_by_y_class_label(configuration_number,data_size,simulation_directory_path,y_class_label_items[1])
    ((dataset_mlbt_x_train,dataset_mlbt_y_train),(dataset_mlbt_x_test,dataset_mlbt_y_test))=dataset_mlbt


# In[ ]:


from jet_ml_dataset_builder_utilities import construct_proportional_y_column
slice_total=10
slice_train=9

print ("Constructing proportional alpha_s y column for y_test and y_train for both MATTER and LBT datasets")
column_values=[alpha_s]
(y_train_alpha_s,y_test_alpha_s)=construct_proportional_y_column(slice_total,slice_train,column_values, data_size)
print("y_train_alpha_s",y_train_alpha_s[0:100])
print("y_test_alpha_s",y_test_alpha_s[0:100])

print ("Constructing proportional q0 y column for y_test and y_train for both MATTER dataset")
column_values=[1]
(y_train_q0_mvac,y_test_q0_mvac)=construct_proportional_y_column(slice_total,slice_train,column_values, data_size)
print("y_train_q0",y_train_q0_mvac[0:100])
print("y_test_q0",y_test_q0_mvac[0:100])

print ("Constructing proportional q0 y column for y_test and y_train for both LBT dataset")
column_values=[q0]
(y_train_q0_mlbt,y_test_q0_mlbt)=construct_proportional_y_column(slice_total,slice_train,column_values, data_size)
print("y_train_q0",y_train_q0_mlbt[0:100])
print("y_test_q0",y_test_q0_mlbt[0:100])


# In[ ]:


from jet_ml_dataset_builder_utilities import concatenate_y_columns_into_dataset

print("Concatenating y_columns (Eloss Module Label, alpha_s, and Q0) into MATTER dataset")
((dataset_mvac_x_train,dataset_mvac_y_train),(dataset_mvac_x_test,dataset_mvac_y_test))=concatenate_y_columns_into_dataset(dataset_mvac,(y_train_alpha_s,y_test_alpha_s),(y_train_q0_mvac,y_test_q0_mvac))
print("dataset_mvac.x_train:",type(dataset_mvac_x_train), dataset_mvac_x_train.size, dataset_mvac_x_train.shape)
print("dataset_mvac.x_test:",type(dataset_mvac_x_test), dataset_mvac_x_test.size, dataset_mvac_x_test.shape)
print("dataset_mvac.y_train:",type(dataset_mvac_y_train), dataset_mvac_y_train.size,dataset_mvac_y_train.shape)
print("dataset_mvac.y_test:",type(dataset_mvac_y_test), dataset_mvac_y_test.size, dataset_mvac_y_test.shape)

print("Concatenating y_columns (Eloss Module Label, alpha_s, and Q0) into LBT dataset")
((dataset_mlbt_x_train,dataset_lbt_y_train),(dataset_lbt_x_test,datasetlbt_y_test))=concatenate_y_columns_into_dataset(dataset_mlbt,(y_train_alpha_s,y_test_alpha_s),(y_train_q0_mlbt,y_test_q0_mlbt))
print("dataset_mlbt.x_train:",type(dataset_mlbt_x_train), dataset_mlbt_x_train.size, dataset_mlbt_x_train.shape)
print("dataset_lbt.x_test:",type(dataset_lbt_x_test), dataset_lbt_x_test.size, dataset_lbt_x_test.shape)
print("dataset_lbt.y_train:",type(dataset_lbt_y_train), dataset_lbt_y_train.size,dataset_lbt_y_train.shape)
print("datasetlbt.y_test:",type(datasetlbt_y_test), datasetlbt_y_test.size, datasetlbt_y_test.shape)


# In[ ]:


from jet_ml_dataset_builder_utilities import store_into_dataset_file

print ("Storing the concaternated dataset with y_columns (Eloss Module Label, alpha_s, and Q0) into file: MATTER")
dataset_mvac=((dataset_mvac_x_train,dataset_mvac_y_train),(dataset_mvac_x_test,dataset_mvac_y_test))
store_into_dataset_file(configuration_number,y_class_label_items[0],data_size,simulation_directory_path,alpha_s,1,dataset_mvac)

print ("Storing the concaternated dataset with y_columns (Eloss Module Label, alpha_s, and Q0) into file: LBT")
dataset_mvac=((dataset_mlbt_x_train,dataset_lbt_y_train),(dataset_lbt_x_test,datasetlbt_y_test))
store_into_dataset_file(configuration_number,y_class_label_items[1],data_size,simulation_directory_path,alpha_s,q0,dataset_mlbt)


# In[ ]:


from jet_ml_dataset_builder_utilities import load_dataset_by_y_class_label
print ("Test: Loading dataset MATTER side and Printing Y side to make sure, 3 colums are")

print("Loading dataset_mvac")
dataset_mvac=None
if dataset_mvac is None:
    dataset_mvac=load_dataset_by_y_class_label(configuration_number,data_size,simulation_directory_path,y_class_label_items[0],alpha_s=0.4,q0=1)
    ((dataset_mvac_x_train,dataset_mvac_y_train),(dataset_mvac_x_test,dataset_mvac_y_test))=dataset_mvac

print("Loading dataset_mlbt")
dataset_mlbt=None
if dataset_mlbt is None:
    dataset_mlbt=load_dataset_by_y_class_label(configuration_number,data_size,simulation_directory_path,y_class_label_items[1],alpha_s=0.4,q0=2.5)
    ((dataset_mlbt_x_train,dataset_mlbt_y_train),(dataset_mlbt_x_test,dataset_mlbt_y_test))=dataset_mlbt

