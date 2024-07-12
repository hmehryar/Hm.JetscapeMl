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


# In[ ]:


def save_event_items_chunk(file_name,event_items_chunks_item):
    with open(file_name, 'wb') as dataset_file:
        pickle.dump(event_items_chunks_item,dataset_file, protocol=pickle.HIGHEST_PROTOCOL)

        
def load__event_items_chunk(file_name):
    with open(file_name, 'rb') as dataset_file:
        event_items = pickle.load(dataset_file, encoding='latin1')
        return event_items


# Loading Events Image Item Chunck Item from Fies and Merge into one file
# 

# In[ ]:


image_grid_count=32
def load_merge_image_items_into_singular_image_file():
  print('\n########################################################################')
  start = time()
  print("Loading Events Image Chunks from File and Merge Into ")
  # number_of_partition=20
  
  number_of_events_per_partition=int( data_size/number_of_partition)
  event_items_image_array =np.array(np.zeros((1,image_grid_count,image_grid_count))) 
  event_items_image = np.array(np.zeros((1,image_grid_count,image_grid_count))) 
  for partition_index in range(number_of_partition):
    print("Partition#",partition_index," Loading Partition ",partition_index, "in the file")
    file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+"-img-chunk.pkl"
    file_name=simulation_directory_path+file_name
    # file_name=y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+"-img-chunk.pkl"
    # file_name=simulation_directory_path+file_name
    event_items_image=load__event_items_chunk(file_name)
    print ("Partition#",partition_index," Loaded: ",file_name)
    print("Partition#",partition_index," event_items_chunks_item type: ", type(event_items_image))
    print("Partition#",partition_index," event_items_chunks_item content: ", len(event_items_image))
    
    print("Partition#",partition_index," Before Appending the new file to events array: ", len(event_items_image_array))
    if partition_index==0:
        event_items_image_array=event_items_image
    else:
        event_items_image_array=np.insert(event_items_image_array,0,event_items_image,axis=0)
    print("Partition#",partition_index," After Appending the new file to events array: ", len(event_items_image_array))
  else:
    print("Finally loading events items partitioning is done!")
  print('\n########################################################################')
  print("Storing Image Items Chunck into File begins")
  file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-img-chunk.pkl"
  file_name=simulation_directory_path+file_name
  # file_name=y_class_label_items[0]+"-simulationsize"+str(data_size)+"-img-chunk.pkl"
  # file_name=simulation_directory_path+file_name
  print ("Dataset Filename: ",file_name,)
  save_event_items_chunk(file_name,event_items_image_array)
  print ("Dataset Stored at : ",file_name)
  print("Storing Image Items Chunck into File ends")
  print('\n########################################################################')
  elapsed = time() - start
  print('Loading / Merging /Storing event image chunk into Singular Elapsed %.3f seconds.' % elapsed)
  print('\n########################################################################')

load_merge_image_items_into_singular_image_file()


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


def adding_missing_images():
    print('\n########################################################################')
    start = time()
    print ("Loading Lbt dataset")
    file_name=y_class_label_items[0]+"-simulationsize"+str(data_size)+"-img-chunk.pkl"
    file_name=simulation_directory_path+file_name
    event_items_image=load__event_items_chunk(file_name)
    print ("Loaded: ",file_name)
    print("Dataset type: ", type(event_items_image))
    print("Dataset length: ", len(event_items_image))

    print ("Loading Lbt dataset")
    file_name="jetscape-ml-benchmark-dataset-matter-vs-lbt-200k-shuffled-01.pkl"
    file_name=dataset_directory_path+file_name
    (x_train, y_train), (x_test, y_test) =load_dataset(file_name)
    print ("Loaded: ",file_name)
    print("Dataset type: ", type(x_train))
    print("Dataset length: ", len(x_train))

    event_items_image_array =np.array(np.zeros((1,image_grid_count,image_grid_count))) 
    print("Before Appending the new file events to array: ", len(event_items_image_array))
    train_index=0
    count=0
    while count!=9:
        if y_train[train_index]==y_class_label_items[0]:
            if count==0:
                event_items_image_array=[x_train[train_index]]
            else:
                event_items_image_array=np.insert(event_items_image_array,0,x_train[train_index],axis=0)
            count=count+1
            print("Label ", y_train[train_index])
            print(len(event_items_image_array))
        train_index=train_index+1          
    print("After Appending the new events to array: ", len(event_items_image_array))
    print("After Appending the new events to array: ", event_items_image_array.size)
    print("After Appending the new events to array: ",type(event_items_image_array)) 
    print(event_items_image_array)  

    print("Appending 9 missing events to dataset") 
    print ("Before Appending size:",len(event_items_image))
    event_items_image=np.insert(event_items_image,0,event_items_image_array,axis=0)
    print ("After Appending size:",len(event_items_image))

    print('\n########################################################################')
    print("Storing Image Items Chunck into File begins")
    file_name=y_class_label_items[0]+"-simulationsize"+str(data_size)+"-img-chunk1.pkl"
    file_name=simulation_directory_path+file_name
    print ("Dataset Filename: ",file_name,)
    save_event_items_chunk(file_name,event_items_image)
    print ("Dataset Stored at : ",file_name)
    print("Storing Image Items Chunck into File ends")
    print('\n########################################################################')
    
    elapsed = time() - start
    print('Loading Dataset Elapsed %.3f seconds.' % elapsed)
    print('\n########################################################################')
# adding_missing_images()


# **Construncting proportional train and test sets** by events' images

# In[ ]:


def get_data_splitting_index(slice_total,slice_train,data_size):
    slice_test=slice_total-slice_train
    data_splitting_index=int(data_size*(slice_train/slice_total))
    return data_splitting_index
def get_x_train_test_data_by_proportion(slice_total,slice_train,data):
    data_size=len(data)
    data_splitting_index=get_data_splitting_index(slice_total,slice_train,data_size)
    x_train=data[0:data_splitting_index]
    x_test=data[data_splitting_index:data_size]
    return (x_train,x_test)
# slice_total=5
# slice_train=4

slice_total=10
slice_train=9

# print('\n########################################################################')
# print('Construncting proportional train and test sets by events'' images')
# (x_train,x_test)=get_x_train_test_data_by_proportion(slice_total,slice_train,event_items_image)
# # (x_train_matter,x_test_matter)=get_x_train_test_data_by_proportion(slice_total,slice_train,event_items_image_matter)
# # (x_train_matter_lbt,x_test_matter_lbt)=get_x_train_test_data_by_proportion(slice_total,slice_train,event_items_image_matter_lbt)


# print("x_train:",type(x_train), x_train.size, x_train.shape)
# print("x_test:",type(x_test), x_test.size, x_test.shape)
# # print("x_train_matter:",type(x_train_matter), x_train_matter.size, x_train_matter.shape)
# # print("x_test_matter:",type(x_test_matter), x_test_matter.size, x_test_matter.shape)
# # print("x_train_matter_lbt:",type(x_train_matter_lbt), x_train_matter_lbt.size, x_train_matter_lbt.shape)
# # print("x_test_matter_lbt:",type(x_test_matter_lbt), x_test_matter_lbt.size, x_test_matter_lbt.shape)
# print('\n########################################################################')


# **Concatenate list of prortional data and create X side of the dataset**

# In[ ]:


def load_construct_proportional_x_sets():
    print('\n########################################################################')
    start = time()
    print ("Loading X dataset")
    file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-img-chunk.pkl"
    file_name=simulation_directory_path+file_name
    event_items_image=load__event_items_chunk(file_name)
    
    print ("Loaded: ",file_name)
    print("Dataset type: ", type(event_items_image))
    print("Dataset length: ", len(event_items_image))
    print('\n########################################################################')
    print('Construncting proportional train and test sets by events'' images')
    (x_train,x_test)=get_x_train_test_data_by_proportion(slice_total,slice_train,event_items_image)
    print("x_train:",type(x_train), x_train.size, x_train.shape)
    print("x_test:",type(x_test), x_test.size, x_test.shape)
    print('\n########################################################################')
    elapsed = time() - start
    print('Loading/Proportionalizing x Dataset Elapsed %.3f seconds.' % elapsed)
    print('\n########################################################################')
    return (x_train,x_test)
(x_train,x_test)=load_construct_proportional_x_sets()


print("x_train:",type(x_train), x_train.size, x_train.shape)
print("x_test:",type(x_test), x_test.size, x_test.shape)


# **Building Y side of the dataset**

# In[ ]:


def dataset_y_builder(y_size,y_class_label_items):
    class_size=int(y_size/len(y_class_label_items))
    y=[]
    for class_label_item in y_class_label_items:
        y = np.append (y, [class_label_item]*class_size)
    return y

def build_y_train_test_data_by_proportion(slice_total,slice_train,y_class_label_items, data_size):
    train_size=get_data_splitting_index(slice_total,slice_train,data_size)
    test_size=data_size-train_size
    y_train=dataset_y_builder(train_size,y_class_label_items)
    y_test=dataset_y_builder(test_size,y_class_label_items)
    return (y_train,y_test)


# data_size=2000
# y_class_label_items=['MVAC','MLBT']
def construct_proportional_y_set():
    print('\n########################################################################')
    start = time()
    print('Building Y side of the dataset')
    (y_train,y_test)=build_y_train_test_data_by_proportion(slice_total,slice_train,y_class_label_items, data_size)
    print("y_train:",type(y_train), y_train.size, y_train.shape)
    print("y_test:",type(y_test), y_test.size, y_test.shape)
    print('\n########################################################################')
    elapsed = time() - start
    print('Proportionalizing y Dataset Elapsed %.3f seconds.' % elapsed)
    print('\n########################################################################')
    return (y_train,y_test)
(y_train,y_test)=construct_proportional_y_set()


# **Saving Constructed Benchmark Dataset as a file**

# In[ ]:


def store_into_dataset_file():
    file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-dataset.pkl"
    file_name=simulation_directory_path+file_name
    # file_name=y_class_label_items[0]+"-simulationsize"+str(data_size)+"-dataset.pkl"
    # file_name=simulation_directory_path+file_name

    print('\n########################################################################')
    print('Saving Constructed Benchmark Dataset as a file')
    dataset=((x_train,y_train),(x_test,y_test))
    save_dataset(file_name,dataset)
    print('\n########################################################################')
store_into_dataset_file()


# In[ ]:


sys.stdout.close()

