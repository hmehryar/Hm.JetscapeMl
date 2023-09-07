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
  dataset_directory_path='/content/drive/MyDrive/Projects/110_JetscapeMl/hm.jetscapeml.data/simulation_results/'
  simulation_directory_path=dataset_directory_path+'simulation-results-dataset-samples/'
elif 'Linux' in running_os:
  dataset_directory_path='/wsu/home/gy/gy40/gy4065/hm.jetscapeml.data/simulation_results/'
  simulation_directory_path=dataset_directory_path+'simulation-results-dataset-samples/'
else:
  dataset_directory_path= 'G:\\My Drive\\Projects\\110_JetscapeMl\\hm.jetscapeml.data\\simulation_results\\'
  simulation_directory_path=dataset_directory_path+'simulation-results-dataset-samples\\'
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
options = "hi:d:y:o:n:c:p:q:a:"
 
# Long options
long_options = ["Help", "Input_file_name_hadrons","Data_size","Y_class_label_items","output_dataset_file_name=", "number_of_partition","configuration_directory","configuration_number","q_0","alpha_s"]
 
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
        elif currentArgument in ("-q", "--q_0"):
            print ("q_0: ", currentValue) 
            q_0=float(currentValue)
            print('q_0 virtuality separation:  {} '.format(q_0))
        elif currentArgument in ("-a", "--alpha_s"):
            print ("alpha_s: ", currentValue) 
            alpha_s=float(currentValue)
            print('alpha_s:  {} '.format(alpha_s))
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
file_name=dataset_directory_path+file_name
event_items_chunks_item=load_event_items_chunk(file_name)
print("event_items_chunks_item type: ", type(event_items_chunks_item))
print("event_items_chunks_item content: ", len(event_items_chunks_item))
# else:
#   print("Finally loading events items partitioning is done!")

print('\n########################################################################')


# Adding required packages and constants

# In[ ]:



from matplotlib.pyplot import figure
from matplotlib.ticker import LogFormatter 
from matplotlib import pyplot as plt, colors
from matplotlib.transforms import offset_copy

#input
pi=3.14159
bin_count=32


# In[ ]:



def convert_event_to_image_with_dark_bg(bin_count,event_item,draw_plot=False):
    event_v = np.vstack(event_item)
    counts, xedges, yedges = np.histogram2d(event_v[:,0], event_v[:,1], bins=bin_count, weights=event_v[:,2])
    
    if draw_plot:

        plt.imshow(counts, interpolation='nearest', origin='lower',
            extent=[-pi, pi, -pi, pi])
        cb = plt.colorbar()
        cb.set_label("density")
        print ("Saving Image: Begin")
        file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+"-single-sample-event-darkbg.png"
        print ("Saving Image: End")
        file_path=simulation_directory_path+file_name
        plt.savefig(file_path)
        plt.show()
        plt.close()
    return counts




# event_item_sample=event_items_chunks_item[0]
# event_item_sample_image=convert_event_to_image_with_dark_bg(bin_count,event_item_sample,True)
# print(type(event_item_sample_image), event_item_sample_image.size, event_item_sample_image.shape)
# print(np.max(event_item_sample))
# print(np.max(event_item_sample_image))


# In[ ]:


def convert_event_to_image(bin_count,event_item,draw_plot=False):
    event_v = np.vstack(event_item)
    fig, ax = plt.subplots()
    counts, xedges, yedges, image = ax.hist2d(event_v[:,0], event_v[:,1],
     bins=bin_count, norm=colors.LogNorm(), weights=event_v[:,2], cmap = plt.cm.jet)
 
    if draw_plot:
        # plt.rcParams["figure.autolayout"] = True
        fig.colorbar(image, ax=ax)
        ax.set_xticks(np.arange(-3, pi, 1)) 
        ax.set_yticks(np.arange(-3, pi, 1)) 
        print ("Saving Image: Begin")
        file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+"-single-sample-event.png"
        # file_name='hm_jetscape_ml_plot_hist_with_white_bg.png'
        file_path=simulation_directory_path+file_name
        plt.savefig(file_path)
        print ("Saving Image: End")
        file_path=simulation_directory_path+file_name
        plt.savefig(file_path)
        plt.show()
        plt.close()
        
    return counts

# event_item_sample=event_items_chunks_item[0] 
# event_item_sample_image=convert_event_to_image(bin_count,event_item_sample,True)
# print(type(event_item_sample_image), event_item_sample_image.size, event_item_sample_image.shape)
# print(np.max(event_item_sample))
# print(np.max(event_item_sample_image))


# In[ ]:


def plot_20_sample_events_with_dark_bg(events_matrix_items):
  # plt.rcParams["figure.autolayout"] = True
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

  file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+"-sample-events-with-dark-bg.png"
  file_path=simulation_directory_path+file_name
  plt.savefig(file_path)

  plt.show()
  plt.close()

#Plotting 20 Sample Events Phase  from shuffled dataset
# events_matrix_items=event_items_chunks_item[0:20]
# plot_20_sample_events_with_dark_bg(events_matrix_items)


# In[ ]:


def plot_sample_events(events_matrix_items):
  plt.rcParams['text.usetex'] = True
  # plt.rcParams["figure.autolayout"] = True
  # fig, axes = plt.subplots(2, 10, figsize=[70,10], dpi=200)
  fig, axes = plt.subplots(2, 5, figsize=[30,10], dpi=200)
  
  # fig.subplots_adjust(top=0.8)

  # Set titles for the figure 
  if y_class_label_items[0]=='MMAT':
    matter_lbt_str='MATTER, In Medium'
  else:
    matter_lbt_str='MATTER+LBT, In Medium'
  suptitle=r'$Config No.{0}: {1}, Q_{{0}}=  {2}, \alpha_{{s}}= {3}$  '.format(configuration_number,matter_lbt_str,q_0,alpha_s)
 
  fig.suptitle(suptitle, fontsize=20, fontweight='bold')
  plt.setp(axes.flat, xlabel=r'$x$', ylabel=r'$y$')
 
  for i, ax in enumerate(axes.flat):
      event_v = np.vstack(events_matrix_items[i])
      
      counts, xedges, yedges, image = ax.hist2d(event_v[:,0], event_v[:,1],
      bins=bin_count, 
      norm=colors.LogNorm(),
      weights=event_v[:,2], cmap = plt.cm.jet)
     
      
      plt.colorbar(image,ax=ax, cmap=cm.jet)
      ax.set_xticks(np.arange(-3, pi, 1)) 
      ax.set_yticks(np.arange(-3, pi, 1))
      # Set titles for the subplot
      ax.set_title(r'Sample {}'.format(i+1))
  

  # cols = ['Sample #{}'.format(col+1) for col in range(0, 10)]
  # rows = ['{}(Medium)'.format(row) for row in ['MATTER', 'MATTER+LBT']]

  # pad = 5 # in points

  # for ax, col in zip(axes[0], cols):
  #     ax.annotate(str(col), xy=(0.5, 1), xytext=(0, pad),
  #                 xycoords='axes fraction', textcoords='offset points',
  #                 size='large', ha='center', va='baseline')

  # for ax, row in zip(axes[:,0], rows):
  #     ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
  #                 xycoords=ax.yaxis.label, textcoords='offset points',
  #                 size='20', ha='right', va='center')

 
  # fig.tight_layout()
  # tight_layout doesn't take these labels into account. We'll need 
  # to make some room. These numbers are are manually tweaked. 
  # You could automatically calculate them, but it's a pain.
  # fig.subplots_adjust(left=0.15, top=0.95)
  file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+"-q0-"+str(q_0)+"-alphas-"+str(alpha_s)+"-sample-events.png"
  file_path=simulation_directory_path+file_name
  plt.savefig(file_path)

  plt.show()
  plt.close()

#Plotting 20 Sample Events Phase  from shuffled dataset
events_matrix_items=event_items_chunks_item[0:10]
plot_sample_events(events_matrix_items)


# In[ ]:


def simple_plot_with_latex():
    plt.rcParams['text.usetex'] = True
    t = np.linspace(0.0, 1.0, 100)
    s = np.cos(4 * np.pi * t) + 2

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.plot(t, s)

    ax.set_xlabel(r'\textbf{time (s)}')
    ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=16)
    ax.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
                r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')
    file_name="sample-equation"
    file_path=simulation_directory_path+file_name
    plt.savefig(file_path) 
# simple_plot_with_latex()    


# In[ ]:


def complex_plot_with_latex():
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots()
        # interface tracking profiles
        N = 500
        delta = 0.6
        X = np.linspace(-1, 1, N)
        ax.plot(X, (1 - np.tanh(4 * X / delta)) / 2,    # phase field tanh profiles
                X, (1.4 + np.tanh(4 * X / delta)) / 4, "C2",  # composition profile
                X, X < 0, "k--")                        # sharp interface

        # legend
        ax.legend(("phase field", "level set", "sharp interface"),
                shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=16)

        # the arrow
        ax.annotate("", xy=(-delta / 2., 0.1), xytext=(delta / 2., 0.1),
                arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
        ax.text(0, 0.1, r"$\delta$",
                color="black", fontsize=24,
                horizontalalignment="center", verticalalignment="center",
                bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2))

        # Use tex in labels
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["$-1$", r"$\pm 0$", "$+1$"], color="k", size=20)

        # Left Y-axis labels, combine math mode and text mode
        ax.set_ylabel(r"\bf{phase field} $\phi$", color="C0", fontsize=20)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([r"\bf{0}", r"\bf{.5}", r"\bf{1}"], color="k", size=20)

        # Right Y-axis labels
        ax.text(1.02, 0.5, r"\bf{level set} $\phi$",
                color="C2", fontsize=20, rotation=90,
                horizontalalignment="left", verticalalignment="center",
                clip_on=False, transform=ax.transAxes)

        # Use multiline environment inside a `text`.
        # level set equations
        eq1 = (r"\begin{eqnarray*}"
        r"|\nabla\phi| &=& 1,\\"
        r"\frac{\partial \phi}{\partial t} + U|\nabla \phi| &=& 0 "
        r"\end{eqnarray*}")
        ax.text(1, 0.9, eq1, color="C2", fontsize=18,
                horizontalalignment="right", verticalalignment="top")

        # phase field equations
        eq2 = (r"\begin{eqnarray*}"
        r"\mathcal{F} &=& \int f\left( \phi, c \right) dV, \\ "
        r"\frac{ \partial \phi } { \partial t } &=& -M_{ \phi } "
        r"\frac{ \delta \mathcal{F} } { \delta \phi }"
        r"\end{eqnarray*}")
        ax.text(0.18, 0.18, eq2, color="C0", fontsize=16)

        ax.text(-1, .30, r"gamma: $\gamma$", color="r", fontsize=20)
        ax.text(-1, .18, r"Omega: $\Omega$", color="b", fontsize=20)

        plt.show()
        file_name="complex-equation"
        file_path=simulation_directory_path+file_name
        plt.savefig(file_path)

# complex_plot_with_latex()

