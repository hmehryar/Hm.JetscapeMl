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




from jet_ml_dataset_builder_utilities import add_alpha_s_and_q0_to_dataset
print ("alpha_s:",alpha_s)
print ("q0:",q0)

print("Adding alpha_s and q to dataset_mvac")

dataset_mvac=add_alpha_s_and_q0_to_dataset(dataset_mvac,alpha_s,1)
((dataset_mvac_x_train,dataset_mvac_y_train),(dataset_mvac_x_test,dataset_mvac_y_test))=dataset_mvac

print("Adding alpha_s and q to dataset_mlbt")
dataset_mlbt=add_alpha_s_and_q0_to_dataset(dataset_mlbt,alpha_s,q0)
((dataset_mlbt_x_train,dataset_mlbt_y_train),(dataset_mlbt_x_test,dataset_mlbt_y_test))=dataset_mlbt



# In[ ]:


from jet_ml_dataset_builder_utilities import store_into_dataset_file

print ("Storing the concaternated dataset with y_columns (Eloss Module Label, alpha_s, and Q0) into file: MATTER")
dataset_mvac=((dataset_mvac_x_train,dataset_mvac_y_train),(dataset_mvac_x_test,dataset_mvac_y_test))
store_into_dataset_file(configuration_number,y_class_label_items[0],data_size,simulation_directory_path,alpha_s,1,dataset_mvac)

print ("Storing the concaternated dataset with y_columns (Eloss Module Label, alpha_s, and Q0) into file: LBT")
dataset_mvac=((dataset_mlbt_x_train,dataset_mlbt_y_train),(dataset_mlbt_x_test,dataset_mlbt_y_test))
store_into_dataset_file(configuration_number,y_class_label_items[1],data_size,simulation_directory_path,alpha_s,q0,dataset_mlbt)


# In[ ]:


from jet_ml_dataset_builder_utilities import load_dataset_by_y_class_label
print ("Test: Loading dataset MATTER side and Printing Y side to make sure, 3 colums are")

print("Loading dataset_mvac")
dataset_mvac=None
if dataset_mvac is None:
    dataset_mvac=load_dataset_by_y_class_label(configuration_number,data_size,simulation_directory_path,y_class_label_items[0],alpha_s,q0=1)
    ((dataset_mvac_x_train,dataset_mvac_y_train),(dataset_mvac_x_test,dataset_mvac_y_test))=dataset_mvac

print("Loading dataset_mlbt")
dataset_mlbt=None
if dataset_mlbt is None:
    dataset_mlbt=load_dataset_by_y_class_label(configuration_number,data_size,simulation_directory_path,y_class_label_items[1],alpha_s,q0)
    ((dataset_mlbt_x_train,dataset_mlbt_y_train),(dataset_mlbt_x_test,dataset_mlbt_y_test))=dataset_mlbt

