#!/usr/bin/env python
# coding: utf-8

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

        
def load_event_items_chunk(file_name):
    with open(file_name, 'rb') as dataset_file:
        event_items = pickle.load(dataset_file, encoding='latin1')
        return event_items


# In[ ]:


print('\n########################################################################')
print("Loading the events chunks from small files")

number_of_events_per_partition=int( data_size/number_of_partition)

print("Loaing Event Items Chunck in multple Array stream")

# for partition_index in range(number_of_partition):
partition_index=0
print("Loading Partition ",partition_index, "in the file")
file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+".pkl"
file_name=simulation_directory_path+file_name
event_items_chunks_item=load_event_items_chunk(file_name)
print("event_items_chunks_item type: ", type(event_items_chunks_item))
print("event_items_chunks_item content: ", len(event_items_chunks_item))
# else:
#   print("Finally loading events items partitioning is done!")

print('\n########################################################################')


# In[ ]:



#input
pi=3.14159

def convert_event_to_image(bin_count,event_item,draw_plot=False):
    event_v = np.vstack(event_item)
    # print (event_v)
    counts, xedges, yedges = np.histogram2d(event_v[:,0], event_v[:,1], bins=bin_count, weights=event_v[:,2])
    # 
    if draw_plot:
#        plt.imshow(counts, interpolation='nearest', origin='lower',
#            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.imshow(counts, interpolation='nearest', origin='lower',
            extent=[-pi, pi, -pi, pi])
        cb = plt.colorbar()
        cb.set_label("density")
        print ("Saving Image: Begin")
        file_name='jet-ml-dataset-histogram-sampler.png'
        print ("Saving Image: End")
        file_path=simulation_directory_path+file_name
        plt.savefig(file_path)
        plt.show()
        plt.close()
    #print(counts)
    return counts

bin_count=32
event_item_sample=event_items_chunks_item[0]


# event_item_sample_image=convert_event_to_image(bin_count,event_item_sample,True)
# print(type(event_item_sample_image), event_item_sample_image.size, event_item_sample_image.shape)
# print(np.max(event_item_sample))
# print(np.max(event_item_sample_image))


# In[ ]:


from matplotlib.pyplot import figure
from matplotlib.ticker import LogFormatter 
from matplotlib import pyplot as plt, colors

def convert_event_to_image_with_white_bg(bin_count,event_item,draw_plot=False):
    event_v = np.vstack(event_item)
    fig, ax = plt.subplots()
    counts, xedges, yedges, image = ax.hist2d(event_v[:,0], event_v[:,1],
     bins=bin_count, norm=colors.LogNorm(), weights=event_v[:,2], cmap = plt.cm.jet)
 
    if draw_plot:
        plt.rcParams["figure.autolayout"] = True
        fig.colorbar(image, ax=ax)
        ax.set_xticks(np.arange(-3, pi, 1)) 
        ax.set_yticks(np.arange(-3, pi, 1)) 
        print ("Saving Image: Begin")
        file_name='hm_jetscape_ml_plot_hist_with_white_bg.png'
        file_path=simulation_directory_path+file_name
        plt.savefig(file_path)
        print ("Saving Image: End")
        file_path=simulation_directory_path+file_name
        plt.savefig(file_path)
        plt.show()
        plt.close()
        
    return counts

bin_count=32
event_item_sample=event_items_chunks_item[0] 
event_item_sample_image=convert_event_to_image_with_white_bg(bin_count,event_item_sample,True)
print(type(event_item_sample_image), event_item_sample_image.size, event_item_sample_image.shape)
print(np.max(event_item_sample))
print(np.max(event_item_sample_image))


# In[ ]:


from matplotlib.pyplot import figure
from matplotlib.ticker import LogFormatter 
from matplotlib import pyplot as plt, colors

def plot_20_sample_events(events_matrix_items):
  plt.rcParams["figure.autolayout"] = True
  fig, axes = plt.subplots(2, 10, figsize=[70,10], dpi=100)
  # fig.text(0.5, 0.04, 'Sample Events Common X', ha='center')
  # fig.text(0.04, 0.5, 'Sample Events common Y', va='center', rotation='vertical')
  for i, ax in enumerate(axes.flat):
      event_v = np.vstack(events_matrix_items[i])
      counts, xedges, yedges = np.histogram2d(event_v[:,0], event_v[:,1], bins=bin_count, weights=event_v[:,2],
      # normed= colors.LogNorm()
      )
      current_plot= ax.imshow(counts, interpolation='nearest', origin='lower',extent=[-pi, pi, -pi, pi])
    
      plt.colorbar(current_plot,ax=ax, cmap=cm.jet)
      ax.set_xticks(np.arange(-3, pi, 1)) 
      ax.set_yticks(np.arange(-3, pi, 1))

  file_name='hm_jetscape_ml_plot_20_sample_events.png'
  file_path=simulation_directory_path+file_name
  plt.savefig(file_path)

  plt.show()
  plt.close()
#Plotting 20 Sample Events Phase  from shuffled dataset

events_matrix_items=event_items_chunks_item[0:20]

# plot_20_sample_events(events_matrix_items)


# In[ ]:


from matplotlib.pyplot import figure
from matplotlib.ticker import LogFormatter 
from matplotlib import pyplot as plt, colors

def plot_20_sample_events_with_white_bg(events_matrix_items):

  plt.rcParams["figure.autolayout"] = True
  fig, axes = plt.subplots(2, 10, figsize=[70,10], dpi=100)
  # fig.text(0.5, 0.04, 'Sample Events Common X', ha='center')
  # fig.text(0.04, 0.5, 'Sample Events common Y', va='center', rotation='vertical')
  for i, ax in enumerate(axes.flat):
      event_v = np.vstack(events_matrix_items[i])
      
      counts, xedges, yedges, image = ax.hist2d(event_v[:,0], event_v[:,1],
      bins=bin_count, 
      norm=colors.LogNorm(),
      weights=event_v[:,2], cmap = plt.cm.jet)
      # current_plot= ax.imshow(counts, interpolation='none',extent=None)
      # 
      # plt.imshow(counts, origin = "lower", interpolation = "gaussian", aspect='equal')

      # counts, xedges, yedges = np.histogram2d(event_v[:,0], event_v[:,1], bins=bin_count, weights=event_v[:,2])
      # current_plot= ax.imshow(counts, interpolation='none', origin='lower',extent=[-pi, pi, -pi, pi])
      # nearest
      
      plt.colorbar(image,ax=ax, cmap=cm.jet)
      ax.set_xticks(np.arange(-3, pi, 1)) 
      ax.set_yticks(np.arange(-3, pi, 1))

  file_name='hm_jetscape_ml_plot_20_sample_events_with_white_bg.png'
  file_path=simulation_directory_path+file_name
  plt.savefig(file_path)

  plt.show()
  plt.close()
#Plotting 20 Sample Events Phase  from shuffled dataset

events_matrix_items=event_items_chunks_item[0:20]

plot_20_sample_events_with_white_bg(events_matrix_items)

