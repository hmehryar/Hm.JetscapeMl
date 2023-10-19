#!/usr/bin/env python
# coding: utf-8

# 
# ## Part 0: Prerequisites:
# 
# 
# 

# In[ ]:


import sys
# sys.stdout = open("output.txt", "w")

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

# if 'google.colab' in str(get_ipython()):
#   print('Running on CoLab')
#   install("google.colab")
#   from google.colab import drive
#   drive.mount('/content/drive')
# else:
#   print('Not running on CoLab')


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


class DatasetBuilderSingleFileAnalyzer:
   # class attribute
  
    # Instance attribute
    def __init__(self, input_file_name_hadrons,data_size,y_class_label_items,output_dataset_file_name):
        self.input_file_name_hadrons=x_train
        self.data_size=y_train
        self.y_class_label_items=x_test
        self.output_dataset_file_name=y_test


# ## Part 0: Input Params:

# getting inputs parameters from command line

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
options = "hi:d:y:o:n:c:p:"
 
# Long options
long_options = ["Help", "Input_file_name_hadrons","Data_size","Y_class_label_items","output_dataset_file_name=", "number_of_partition","configuration_directory","configuration_number"]
 
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
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
print('########################################################################\n')


# ##Setting the input parameters in hardcoded

# In[ ]:


# print('########################################################################\n')

# # file_name_matter='finalStateHadrons-Matter.dat'
# # file_name_matter='finalStateHadrons-Matter-100k.dat'
# file_name_matter='finalStateHadrons-Matter-600k.dat'
# # file_name_matter_lbt='finalStateHadrons-MatterLbt.dat'
# # file_name_matter_lbt='finalStateHadrons-MatterLbt-100k.dat'
# file_name_matter_lbt='finalStateHadrons-MatterLbt-600k.dat'

# file_name_hadrons='finalStateHadrons-Matter-1k.dat'
# print('simulated events final state hadron file: '+file_name_hadrons)

# data_size=1000
# print('data_size: {} '.format(data_size))
# print
# # ['MVAC','MLBT']
# y_class_label_items=['MVAC']
# print("y_class_label_items")
# print(y_class_label_items)

# #dataset_file_name='jetscape-ml-benchmark-dataset-2k-randomized.pkl'
# # dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-2000.pkl'
# # dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-1200k-momentum.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-1k-matter.pkl'

# print("Dataset file name: "+dataset_file_name)
# print('########################################################################\n')


# Loading Events Image Item Chunck Item from Fies and Merge into one file

# In[ ]:


def save_dataset(file_name,dataset):
    with open(file_name, 'wb') as dataset_file:
        pickle.dump(dataset,dataset_file, protocol=pickle.HIGHEST_PROTOCOL)

        
def load_dataset(file_name):
    with open(file_name, 'rb') as dataset_file:
        (x_train, y_train), (x_test, y_test) = pickle.load(dataset_file, encoding='latin1')
        dataset=((x_train, y_train), (x_test, y_test))
        return dataset


# In[ ]:


image_grid_count=32
def load_and_concatenate_datasets():
    print('\n########################################################################')
    print('Loading separate datasets')
    y_class_label_items=['MMAT','MLBT']

    file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-dataset.pkl"
    file_name=simulation_directory_path+file_name
    # file_name=y_class_label_items[0]+"-simulationsize"+str(data_size)+"-dataset.pkl"
    # file_name=simulation_directory_path+file_name
    ((dataset_mvac_x_train,dataset_mvac_y_train),(dataset_mvac_x_test,dataset_mvac_y_test))= load_dataset(file_name)
    print("dataset_mvac.x_train:",type(dataset_mvac_x_train), dataset_mvac_x_train.size, dataset_mvac_x_train.shape)
    print("dataset_mvac.x_test:",type(dataset_mvac_x_test), dataset_mvac_x_test.size, dataset_mvac_x_test.shape)
    print("dataset_mvac.y_train:",type(dataset_mvac_y_train), dataset_mvac_y_train.size,dataset_mvac_y_train.shape)
    print("dataset_mvac.y_test:",type(dataset_mvac_y_test), dataset_mvac_y_test.size, dataset_mvac_y_test.shape)
    file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[1]+"-simulationsize"+str(data_size)+"-dataset.pkl"
    file_name=simulation_directory_path+file_name
    # file_name=y_class_label_items[1]+"-simulationsize"+str(data_size)+"-dataset.pkl"
    # file_name=simulation_directory_path+file_name
    ((dataset_mlbt_x_train,dataset_mlbt_y_train),(dataset_mlbt_x_test,dataset_mlbt_y_test))= load_dataset(file_name)
    print("dataset_mlbt.x_train:",type(dataset_mlbt_x_train), dataset_mlbt_x_train.size, dataset_mlbt_x_train.shape)
    print("dataset_mlbt.x_test:",type(dataset_mlbt_x_test), dataset_mlbt_x_test.size, dataset_mlbt_x_test.shape)
    print("dataset_mlbt.y_train:",type(dataset_mlbt_y_train), dataset_mlbt_y_train.size,dataset_mlbt_y_train.shape)
    print("dataset_mlbt.y_test:",type(dataset_mlbt_y_test), dataset_mlbt_y_test.size, dataset_mlbt_y_test.shape)

    x_train =np.array(np.zeros((1,image_grid_count,image_grid_count))) 
    x_train=dataset_mvac_x_train
    x_train=np.insert(x_train,0,dataset_mlbt_x_train,axis=0)
   
    
    y_train=[]
    y_train= np.append (y_train, dataset_mvac_y_train)
    y_train= np.append (y_train, dataset_mlbt_y_train)
    
    x_test =np.array(np.zeros((1,image_grid_count,image_grid_count))) 
    x_test=dataset_mvac_x_test
    x_test=np.insert(x_test,0,dataset_mlbt_x_test,axis=0)
   

    y_test=[]
    y_test= np.append (y_test, dataset_mvac_y_test)
    y_test= np.append (y_test, dataset_mlbt_y_test)

    print("dataset.x_train:",type(x_train), x_train.size, x_train.shape)
    print("dataset.x_test:",type(x_test), x_test.size, x_test.shape)
    print("dataset.y_train:",type(y_train), y_train.size,y_train.shape)
    print("dataset.y_test:",type(y_test), y_test.size, y_test.shape)
    print('\n########################################################################')
    dataset=((x_train,y_train),(x_test,y_test))
    return dataset

def concatenate_and_store_dataset_into_single_file():
    print('\n########################################################################')
    start = time() 
    dataset=load_and_concatenate_datasets()
    print('\n########################################################################')
    print('Saving Constructed Benchmark Dataset as a file')
    file_name="config-0"+str(configuration_number)+"-matter-vs-lbt-simulationsize"+str(data_size)+"-dataset-momentum.pkl"
    # file_name="jetscape-ml-benchmark-dataset-matter-vs-lbt-1200k-momentum.pkl"
    file_name=simulation_directory_path+file_name
    save_dataset(file_name,dataset)
    print('\n########################################################################')
    elapsed = time() - start
    print('Concatenating and Storing Elapsed %.3f seconds.' % elapsed)
    print('\n########################################################################')

concatenate_and_store_dataset_into_single_file()


# In[ ]:


def shuffle_training_dataset(x_train, y_train):
  
  print("Train Dataset Permutation Array:")
  train_permutation_array_indices=np.random.permutation(y_train.size)
  #print(train_permutation_array_indices[1:100])

  print("y_train:")
  print(y_train, type(y_train),y_train.size, y_train.shape)
  #print(y_train[1:100])

  print("y_train_shuffled:")
  y_train_shuffled=np.take(y_train, train_permutation_array_indices)
  print(y_train_shuffled, type(y_train_shuffled),y_train_shuffled.size, y_train_shuffled.shape)
  #print(y_train_shuffled[1:100])

  print("x_train:")
  print(x_train, type(x_train),x_train.size, x_train.shape)
  #print(x_train[1:100])

  print("x_train_shuffled:")
  x_train_shuffled=np.take(x_train, train_permutation_array_indices,axis=0)
  print(x_train_shuffled, type(x_train_shuffled),x_train_shuffled.size, x_train_shuffled.shape)
  #print(x_train_shuffled[1:100])

  dataset_train_shuffled=(x_train_shuffled, y_train_shuffled)
  return dataset_train_shuffled


#main method

def shuffle_training_dataset_runner():
    print('\n########################################################################')
    start_time = time() 

    # file_name="jetscape-ml-benchmark-dataset-matter-vs-lbt-1200k-momentum.pkl"
    file_name="config-0"+str(configuration_number)+"-matter-vs-lbt-simulationsize"+str(data_size)+"-dataset-momentum.pkl"
    file_name=simulation_directory_path+file_name

    print("Loading Data Set")
    (x_train, y_train), (x_test, y_test) =load_dataset(file_name)

    print("Shuffling Data Set")
    (x_train_shuffled, y_train_shuffled)=shuffle_training_dataset(x_train, y_train)

    
    # file_name="jetscape-ml-benchmark-dataset-matter-vs-lbt-1200k-momentum-shuffled.pkl"
    file_name="config-0"+str(configuration_number)+"-matter-vs-lbt-simulationsize"+str(data_size)+"-dataset-momentum-shuffled.pkl"
    file_name=simulation_directory_path+file_name
    dataset=((x_train_shuffled,y_train_shuffled),(x_test,y_test))
    save_dataset(file_name,dataset)
    end_time=time()

    print('\n########################################################################')
    elapsed = time() - start_time
    print('Shuffling / Storing Elapsed %.3f seconds.' % elapsed)
    print('\n########################################################################')
shuffle_training_dataset_runner()

