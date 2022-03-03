"""
Created on Fri Feb 25 14:02:35 2022

@author: HP
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Commonly used modules
import numpy as np
import os
import sys

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#import cv2
import IPython
from six.moves import urllib

print(tf.__version__)
class jetscapeMl:
  def __init__(self, dataset_size, dataset_is_local):
    self.dataset_size = dataset_size
    self.dataset_is_local = dataset_is_local
    print("JETSCAPE-ML: \n\tDataSetSize: {0}\n\tDataSetIsInLocalMachine: {1}"
          .format(self.dataset_size,self.dataset_is_local))
  
#   **Building Randomized Dataset**
# Before having the simulation data, researcher tried to implmenet a 
# psudo random data to create the architecture for the project. 
# I thought it could be useful for further usages.
  import numpy as np
  
  def dataset_y_builder(self,y_size,y_class_label_items):
      class_size=int(y_size/len(y_class_label_items))
      y=[]
      for class_label_item in y_class_label_items:
          y = np.append (y, [class_label_item]*class_size)
      return y

  def dataset_x_builder_randomized(self,x_size,frame_size):
      x=np.arange(x_size*dataset_frame_size*dataset_frame_size).reshape((x_size,frame_size,frame_size))
      return x

# **Saving Dataset Benchmark as a file**
#!pip3 install pickle5
#  import pickle5 as pickle
  import pickle
#  from google.colab import drive

  def saveDataset(self,file_name,dataset):
      if (not self.dataset_is_local):
          #TODO
          print("TODO: Implementing Google Drive Connection")
          #drive.mount('/content/drive')
      with open(file_name, 'wb') as dataset_file:
          pickle.dump(dataset,dataset_file, protocol=pickle.HIGHEST_PROTOCOL)


#main
oJetscapeMl=jetscapeMl( dataset_size="200K", dataset_is_local=False)
 

# Building Randomized Dataset
dataset_frame_size=32
#train_size=600
#test_size=100
train_size=1600
test_size=400

y_class_label_items=['MVAC','MMED','MLBT','MMAR']
y_train=oJetscapeMl.dataset_y_builder(train_size,y_class_label_items)
y_test=oJetscapeMl.dataset_y_builder(test_size,y_class_label_items)


x_train=oJetscapeMl.dataset_x_builder_randomized(train_size,dataset_frame_size)
x_test=oJetscapeMl.dataset_x_builder_randomized(test_size,dataset_frame_size)

print(type(x_train), x_train.size, x_train.shape)
print(type(y_train), y_train.size, y_train.shape)
print(type(x_test), x_test.size, x_test.shape)
print(type(y_test), y_test.size, y_test.shape)
#End: Building Randomized Dataset

file_directory_path= 'G:\\My Drive\\Projects\\110_JetscapeMl\\Hm.JetscapeMl.Data\\'
#file_directory_path= '/content/drive/MyDrive/Projects/110_JetscapeMl/Hm.JetscapeMl.Data/'

file_name='jetscape-ml-benchmark-dataset-2k-randomized-testing-single-file.pkl'
dataset=((x_train,y_train),(x_test,y_test))
oJetscapeMl.saveDataset(file_directory_path+file_name,dataset)

#(x_train, y_train), (x_test, y_test) = oJetscapeMl.load_data() 