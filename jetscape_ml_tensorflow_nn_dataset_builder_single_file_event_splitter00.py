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
options = "hi:d:y:o:n:"
 
# Long options
long_options = ["Help", "Input_file_name_hadrons","Data_size","Y_class_label_items","output_dataset_file_name=", "number_of_partition"]
 
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


# **Extracting Column 5, and 6 from simulation data**

# In[ ]:




momentum_col_num=3
x_col_num=4
y_col_num=5
#hit_data_items=np.loadtxt(file_name, usecols=(4,5,))
#print(hit_data_items)
def extracting_hit_items(
    file_name,x_col_num,y_col_num,momentum_col_num):
    hit_data_items=np.loadtxt(dataset_directory_path+file_name, usecols=(x_col_num,y_col_num,momentum_col_num))
    return hit_data_items


#main
print('\n########################################################################')
print('extracting_hit_items => Begin\n\n')

print("extracting_hit_items from "+file_name_hadrons)
start = time()

hit_data_items=extracting_hit_items(file_name_hadrons,x_col_num,y_col_num,momentum_col_num)


elapsed = time() - start
print('extracting_hit_items : Elapsed %.3f seconds.' % elapsed)

print(hit_data_items)

print('extracting_hit_items => Ends\n\n')
print('\n########################################################################')


# **Setting the cone size and filtering extra hits**

# In[ ]:


def get_in_range_hit_items(hit_data_items,cone_radius):
    hit_data_in_range=[item for item in hit_data_items if item[0]<cone_radius 
                       and item[0]>-cone_radius 
                       and item[1]<cone_radius 
                       and item[1]>-cone_radius]
    return hit_data_in_range

print('\n########################################################################')
print("hit data size before filtering ",hit_data_items.size)
# print("hit data size before filtering (MATTER)",hit_data_items_matter.size)
# print("hit data size before filtering (MATTER+LBT)",hit_data_items_matter_lbt.size)

cone_radius=3.14159

hit_data_items_in_range=get_in_range_hit_items(hit_data_items,cone_radius)
# hit_data_items_in_range_matter=get_in_range_hit_items(hit_data_items_matter,cone_radius)
# hit_data_items_in_range_matter_lbt=get_in_range_hit_items(hit_data_items_matter_lbt,cone_radius)

print("hit data size after filtering :",len(hit_data_items_in_range))
# print("hit data size after filtering (MATTER):",len(hit_data_items_in_range_matter))
# print("hit data size after filtering (MATTER+LBT):",len(hit_data_items_in_range_matter_lbt))


# hit_data_items=hit_data_items_matter

#hit_data_in_range=[];
#print(hit_data_items.size)
#hit_data_in_range=[item for item in hit_data_items if item[0]<cone_radius and item[0]>-cone_radius and item[1]<cone_radius and item[1]>-cone_radius]
#print(len(hit_data_in_range))
#print(type(hit_data_in_range), type(hit_data_items))
print('\n########################################################################')


# **splitting the data in an array of events**
# This method has implemented  just by checking the zero as the line separator

# In[ ]:




def get_splitted_events_array(hit_data_items):
    events = []
    current_event=[]
    count=0
    for hit_data_item in hit_data_items:
        is_all_zeros = not np.any(hit_data_item)
        if is_all_zeros:
            events.append((current_event))
            count=count+1
            #print(count)
            #print('**************')
            #print(current_event)
            current_event=[]
        else:
            current_event.append((hit_data_item[0],hit_data_item[1],hit_data_item[2]))
    #print("Total number of event items: ",count)
    return events

print('\n########################################################################')
print('Splitting all events into separated events')
event_items=get_splitted_events_array(hit_data_items_in_range)
# event_items_matter=get_splitted_events_array(hit_data_items_in_range_matter)
# event_items_matter_lbt=get_splitted_events_array(hit_data_items_in_range_matter_lbt)

print("number of event items :",len(event_items))
# print("number of event items (MATTER):",len(event_items_matter))
# print("number of event items (MATTER+LBT):",len(event_items_matter_lbt))

#events = []
#current_event=[]
#count=0
#for hit_data_item in hit_data_items:
#    is_all_zeros = not np.any(hit_data_item)
#    if is_all_zeros:
#        events.append((current_event))
#        count=count+1
#        print(count)
#        print('**************')
#        print(current_event)
#        current_event=[]
#    else:
#        current_event.append((hit_data_item[0],hit_data_item[1]))

##mat_vals = np.vstack(events)
##print(events)
##print(len(mat_vals))
#print(len(events))
print('\n########################################################################')


# In[ ]:


def save_event_items_chunk(file_name,event_items_chunks_item):
    with open(file_name, 'wb') as dataset_file:
        pickle.dump(event_items_chunks_item,dataset_file, protocol=pickle.HIGHEST_PROTOCOL)

        
def load__event_items_chunk(file_name):
    with open(file_name, 'rb') as dataset_file:
        event_items = pickle.load(dataset_file, encoding='latin1')
        return event_items


# Storing Events Item Chunck Item into File

# In[ ]:


print('\n########################################################################')
start = time()
print("Splitting the events array to smaller chunck")
# number_of_partition=20
number_of_events_per_partition=int( data_size/number_of_partition)
event_items_chunks_array= np.array_split(event_items, number_of_partition)
print("event_items_chunks_array type: ", type(event_items_chunks_array))
print("event_items_chunks_array content: ", len(event_items_chunks_array))
print("Storing Event Items Chunck in multple Files")

for partition_index in range(number_of_partition):
  print("Storing Partition ",partition_index, "in the file")
  file_name=y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+".pkl"
  file_name=simulation_directory_path+file_name
  print ("Filename: ",file_name)
  save_event_items_chunk(file_name,event_items_chunks_array[partition_index])
  print ("Stored at : ",file_name)
else:
  print("Finally Storing events items partitioning is done!")
elapsed = time() - start
print('Storing Elapsed %.3f seconds.' % elapsed)
print('\n########################################################################')


# Loading Events Item Chunck Item from File

# In[ ]:


print('\n########################################################################')
print("Loading the events chunks from small files")
# number_of_partition=20
number_of_events_per_partition=int( data_size/number_of_partition)

print("Loaing Event Items Chunck in multple Array stream")

for partition_index in range(number_of_partition):
  print("Loading Partition ",partition_index, "in the file")
  file_name=y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+".pkl"
  file_name=simulation_directory_path+file_name
  event_items_chunks_item=load__event_items_chunk(file_name)
  print("event_items_chunks_item type: ", type(event_items_chunks_array))
  print("event_items_chunks_item content: ", len(event_items_chunks_array))
else:
  print("Finally loading events items partitioning is done!")

print('\n########################################################################')


# **Converting event items into 32x32 pixel images/2-D Array**
# and Plotting sample event histogram image of a jet shower event
# 

# In[ ]:



# #input
# pi=3.14159

# def convert_event_to_image(bin_count,event_item,draw_plot=False):
#     event_v = np.vstack(event_item)
#     # print (event_v)
#     counts, xedges, yedges = np.histogram2d(event_v[:,0], event_v[:,1], bins=bin_count, weights=event_v[:,2])
#     # 
#     if draw_plot:
# #        plt.imshow(counts, interpolation='nearest', origin='lower',
# #            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#         plt.imshow(counts, interpolation='nearest', origin='lower',
#             extent=[-pi, pi, -pi, pi])
#         cb = plt.colorbar()
#         cb.set_label("density")
#     #print(counts)
#     return counts

# bin_count=32
# print('\n########################################################################')
# print('Converting event items into 32x32 pixel images/2-D Array and Plotting sample event histogram image of a jet shower event')
# event_item_sample=event_items[0]
# # print(event_item_sample)

# event_item_sample_image=convert_event_to_image(bin_count,event_item_sample,True)
# print(type(event_item_sample_image), event_item_sample_image.size, event_item_sample_image.shape)
# # print(event_item_sample_image)
# print(np.max(event_item_sample))
# print(np.max(event_item_sample_image))
# print('\n########################################################################')


# **Converting all events into image data structure**

# In[ ]:


def convert_events_to_images(image_grid_count,event_items):
    event_items_image = np.array(np.zeros((1,image_grid_count,image_grid_count)))
    is_first_cell_zero=True
    for event_item in event_items:
        event_item_image=convert_event_to_image(image_grid_count,event_item,False)
        if is_first_cell_zero:
            event_items_image[0]=event_item_image
            is_first_cell_zero=False
        else:
            event_items_image=np.insert(event_items_image,0,event_item_image,axis=0)
    return event_items_image
 


# In[ ]:


# # event_items=event_items_matter
# image_grid_count=32

# print('\n########################################################################')
# print('Converting all events into image data structure')
# event_items_image=convert_events_to_images(image_grid_count,event_items)  
# # event_items_image_matter=convert_events_to_images(image_grid_count,event_items_matter)  
# # event_items_image_matter_lbt=convert_events_to_images(image_grid_count,event_items_matter_lbt) 

# print(type(event_items_image), event_items_image.size, event_items_image.shape)
# # print(type(event_items_image_matter), event_items_image_matter.size, event_items_image_matter.shape)
# # print(type(event_items_image_matter_lbt), event_items_image_matter_lbt.size, event_items_image_matter_lbt.shape)
# print('\n########################################################################')


# **Construncting proportional train and test sets** by events' images

# In[ ]:


# def get_data_splitting_index(slice_total,slice_train,data_size):
#     slice_test=slice_total-slice_train
#     data_splitting_index=int(data_size*(slice_train/slice_total))
#     return data_splitting_index
# def get_x_train_test_data_by_proportion(slice_total,slice_train,data):
#     data_size=len(data)
#     data_splitting_index=get_data_splitting_index(slice_total,slice_train,data_size)
#     x_train=data[0:data_splitting_index]
#     x_test=data[data_splitting_index:data_size]
#     return (x_train,x_test)

# # slice_total=5
# # slice_train=4

# slice_total=10
# slice_train=9

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


# def concatenate_x_dataset_by_proportion_items(proportion_items):
#     x_dataset=np.array(np.zeros((1,32,32)))
#     is_first_cell_zero=True
#     for proportion_item in proportion_items:
#         if is_first_cell_zero:
#             x_dataset=proportion_items[0]
#             is_first_cell_zero=False
#         else:
#             x_dataset=np.insert(x_dataset,0,proportion_item,axis=0)
            
#     return x_dataset

# x_train_proportion_items=[x_train_matter,x_train_matter_lbt]
# x_train=concatenate_x_dataset_by_proportion_items(x_train_proportion_items)

# x_test_proportion_items=[x_test_matter,x_test_matter_lbt]
# x_test=concatenate_x_dataset_by_proportion_items(x_test_proportion_items)

# print("x_train:",type(x_train), x_train.size, x_train.shape)
# print("x_test:",type(x_test), x_test.size, x_test.shape)


# **Building Y side of the dataset**

# In[ ]:


# def dataset_y_builder(y_size,y_class_label_items):
#     class_size=int(y_size/len(y_class_label_items))
#     y=[]
#     for class_label_item in y_class_label_items:
#         y = np.append (y, [class_label_item]*class_size)
#     return y

# def build_y_train_test_data_by_proportion(slice_total,slice_train,y_class_label_items, data_size):
#     train_size=get_data_splitting_index(slice_total,slice_train,data_size)
#     test_size=data_size-train_size
#     y_train=dataset_y_builder(train_size,y_class_label_items)
#     y_test=dataset_y_builder(test_size,y_class_label_items)
#     return (y_train,y_test)


# # data_size=2000
# # y_class_label_items=['MVAC','MLBT']
# print('\n########################################################################')
# print('Building Y side of the dataset')
# (y_train,y_test)=build_y_train_test_data_by_proportion(slice_total,slice_train,y_class_label_items, data_size)
# print("y_train:",type(y_train), y_train.size, y_train.shape)
# print("y_test:",type(y_test), y_test.size, y_test.shape)
# print('\n########################################################################')


# **Saving Constructed Benchmark Dataset as a file**

# In[ ]:


# def saveDataset(file_name,dataset):
#     with open(file_name, 'wb') as dataset_file:
#         pickle.dump(dataset,dataset_file, protocol=pickle.HIGHEST_PROTOCOL)
# print('\n########################################################################')
# print('Saving Constructed Benchmark Dataset as a file')
# dataset=((x_train,y_train),(x_test,y_test))
# saveDataset(dataset_directory_path+dataset_file_name,dataset)
# print('\n########################################################################')


# In[ ]:


sys.stdout.close()

