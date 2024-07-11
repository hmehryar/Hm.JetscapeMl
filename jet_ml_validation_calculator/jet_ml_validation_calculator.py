#!/usr/bin/env python
# coding: utf-8

# In[ ]:



print('Loading/Installing Package => Begin\n\n')
# Commonly used modules
import numpy as np
import os
from os import path, makedirs
import time
from time import time
import subprocess
import sys


def install(package):
  print("Installing "+package) 
  subprocess.check_call([sys.executable,"-m" ,"pip", "install", package])
  print("Installed "+package+"\n") 
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import load_model

# !pip3 install sklearn
install("sklearn")
from sklearn.metrics import confusion_matrix, classification_report

# !pip3 install seaborn
install("seaborn")
import seaborn as sns


# Images, plots, display, and visualization
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

#reading/writing into files
# !pip3 install pickle5
install("pickle5")
import pickle5 as pickle


#import cv2
install("IPython")
import IPython
from six.moves import urllib


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
print("Tensorflow version: "+tf.__version__)

dataset_directory_path=''
simulation_directory_path=''

if COLAB == True:
  drive.mount('/content/drive')
  dataset_directory_path='/content/drive/MyDrive/Projects/110_JetscapeMl/hm.jetscapeml.data/simulation_results/'
  simulation_directory_path=dataset_directory_path+'simulation-results-deep-model-cnn-01-1200K-config-05-epoch-30/'
elif 'Linux' in running_os:
  dataset_directory_path='/wsu/home/gy/gy40/gy4065/hm.jetscapeml.data/simulation_results/'
  simulation_directory_path=dataset_directory_path+'simulation-results-deep-model-cnn-01-1200K-config-05-epoch-30/'
else:
  dataset_directory_path= 'G:\\My Drive\\Projects\\110_JetscapeMl\\hm.jetscapeml.data\\simulation_results\\'
  simulation_directory_path=dataset_directory_path+'simulation-results-deep-model-cnn-01-1200K-config-05-epoch-30\\'
print('Dataset Directory Path: '+dataset_directory_path)

#dataset_file_name='jetscape-ml-benchmark-dataset-2k-randomized.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-2000.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-200k-shuffled-01.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-1200k-momentum-shuffled.pkl'
# dataset_file_name='config-01-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-02-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-03-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-04-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
dataset_file_name='config-05-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-06-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-07-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-08-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-09-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
print("Dataset file name: "+dataset_file_name)

if not path.exists(simulation_directory_path):
    makedirs(simulation_directory_path)
print('Simulation Results Path: '+simulation_directory_path)
print('########################################################################\n')
print('\nLoading/Installing Package => End\n\n')


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


class JetscapeMlCnn:
   # class attribute
  
    # Instance attribute
    def __init__(self, x_train,y_train,x_test,y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test


#Loading Dataset Phase


dataset_file_path=dataset_directory_path+dataset_file_name
print("Dataset file path: "+dataset_file_path)
(x_train, y_train), (x_test, y_test) =load_dataset(dataset_file_path)

oJetscapeMlCnn=JetscapeMlCnn(x_train, y_train, x_test, y_test)
print("\n#############################################################")
print("Post-Load: DataType Checkpoint: Begin")
print(type(oJetscapeMlCnn.x_train), oJetscapeMlCnn.x_train.size, oJetscapeMlCnn.x_train.shape)
print(type(oJetscapeMlCnn.y_train), oJetscapeMlCnn.y_train.size, oJetscapeMlCnn.y_train.shape)
print(type(oJetscapeMlCnn.x_test), oJetscapeMlCnn.x_test.size, oJetscapeMlCnn.x_test.shape)
print(type(oJetscapeMlCnn.y_test), oJetscapeMlCnn.y_test.size, oJetscapeMlCnn.y_test.shape)
print(oJetscapeMlCnn.y_train[1500], oJetscapeMlCnn.y_test[99])
print(oJetscapeMlCnn.y_train[1:500])
print("Post-Load: DataType Checkpoint: End")
print("#############################################################\n")


# In[ ]:



def convertDatasetYFromLiteralToNumeric(y_dataset):
  y_train=y_dataset[0]
  y_test=y_dataset[1]
  y_train_unique_class_labels,y_train_positions = np.unique(y_train,return_inverse=True)
  y_test_unique_class_labels,y_test_positions = np.unique(y_test,return_inverse=True)
  
  print(y_train_unique_class_labels)
  print(y_test_unique_class_labels)
  
  y_train=y_train_positions
  y_test=y_test_positions
  
  return ((y_train,y_test))


print("\n#############################################################")
print("Changing classification labels from Literal to Numeric:")
print("\nBefore conversion:")
print(type(y_train), y_train.size, y_train.shape)
print(type(y_test), y_test.size, y_test.shape)
print(type(y_train[0]))
print(type(y_test[0]))

y_train,y_test =convertDatasetYFromLiteralToNumeric((y_train,y_test))

print("\nAfter conversion:")
print(type(y_train), y_train.size, y_train.shape)
print(type(y_test), y_test.size, y_test.shape)

print(type(y_train[0]))
print(type(y_test[0]))
print("#############################################################\n")




# In[ ]:




# Reserve 20% samples for validation dataset
def calculate_validation_dataset_size(dataset_train_size,dataset_test_size):
  dataset_size= dataset_train_size+dataset_test_size
  dataset_validation_size=dataset_size*.2
  return int(dataset_validation_size)

def set_validation_dataset(x_train,y_train,validation_dataset_size):
  
  x_val = x_train[-validation_dataset_size:]
  y_val = y_train[-validation_dataset_size:]
  x_train = x_train[:-validation_dataset_size]
  y_train = y_train[:-validation_dataset_size]
  
  
  return (x_train, y_train), (x_val, y_val)

validation_dataset_size= calculate_validation_dataset_size(y_train.size,y_test.size)
(x_train, y_train), (x_val, y_val)=set_validation_dataset(x_train,y_train,validation_dataset_size)
print("\n#############################################################")
print("Defining Validation Dataset from Train Dataset:")

print("\nTrain data info:")
print(type(y_train), y_train.size, y_train.shape)
print(type(x_train), x_train.size, x_train.shape)

print("\nValidation data info:")
print(type(y_val), y_val.size, y_val.shape)
print(type(x_val), x_val.size, x_val.shape)

print("\nTest data info:")
print(type(y_test), y_test.size, y_test.shape)
print(type(x_test), x_test.size, x_test.shape)
print("#############################################################\n")


# In[ ]:


print("\n#############################################################")
print("Reshaping dataset X-side dimension to be fit in the defined convolutional :")

print("\nX train:")
print(x_train.shape)
print (x_train.shape[0],x_train.shape[1],x_train.shape[2])
x_train_reshaped=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
print(x_train_reshaped.shape)

print("\nX val:")
print(x_val.shape)
print (x_val.shape[0],x_val.shape[1],x_val.shape[2])
x_val_reshaped=x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)
print(x_train_reshaped.shape)

print("\nX test:")
print(x_test.shape)
print (x_test.shape[0],x_test.shape[1],x_test.shape[2])
x_test_reshaped=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print(x_test_reshaped.shape)
print("#############################################################\n")


# In[ ]:


## event info
collision = 'PbPb'
energy = 5020
centrality = '0_10'
Modules = ['MATTER','LBT']
JetptMinMax = '100_110'
#observables = ['pt','charge','mass']
observables = ['pt']
kind = 'Hadron'


# In[ ]:


save_dir = (simulation_directory_path+'models_{}_vs_{}_{}_ch{}').format(Modules[0], Modules[1], kind, len(observables))
if not path.exists(save_dir):
    makedirs(save_dir)
print('Directory to save models: {}'.format(save_dir))



# In[ ]:


file_name='hm_jetscape_ml_model_history.npy'
file_path=simulation_directory_path+file_name


# In[ ]:


# training and validation sets
train_set, val_set = (x_train_reshaped, y_train), (x_val_reshaped, y_val)


# In[ ]:


# This section shall be just used after training or for stand alone evaluations
# Building a dictionary which is accessable by dot
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#Loading learning history after training 
file_name='hm_jetscape_ml_model_history.npy'
file_path=simulation_directory_path+file_name


history=dict({'history':np.load(file_path,allow_pickle='TRUE').item()})
history=dotdict(history)
print(history)


# In[ ]:



from matplotlib import pyplot as plt
def plot_train_history(history):

    color_list = ['red','blue','black','green']

    plt.figure(figsize=(8, 2.5), dpi=200)

    plt.subplot(121)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.title('Loss history')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy history')
    plt.legend()
    file_name='hm_jetscape_ml_plot_train_history.png'
    file_path=simulation_directory_path+file_name
    plt.savefig(file_path)
    plt.show()
    plt.close()
# plot the training history for each fold
plot_train_history(history)


# In[ ]:



from tensorflow.keras.models import load_model
## load the best model
best_model = load_model(path.join(save_dir,'hm_jetscape_ml_model_best.h5'))

outputStr='Train   | Validation | Test sets\n'

## evaluate the model on train/val/test sets and append the results to lists
_, train_acc = best_model.evaluate(x_train_reshaped, y_train, verbose=0)
_, val_acc = best_model.evaluate(x_val_reshaped, y_val, verbose=0)
_, test_acc = best_model.evaluate(x_test_reshaped, y_test, verbose=0)
    
## print out the accuracy
outputStr+='{:.4f}%  {:.4f}%     {:.4f}%\n'.format(train_acc * 100, val_acc * 100, test_acc * 100)
print(outputStr)

file_name="hm_jetscape_ml_model_evaluation.txt"
file_path=simulation_directory_path+file_name
evaluation_file = open(file_path, "w")
evaluation_file.write(outputStr)
evaluation_file.close()


# In[ ]:



from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
## plot confution matrix
y_pred = best_model.predict_classes(x_test_reshaped)

conf_mat = confusion_matrix(y_pred, y_test)
sns.heatmap(conf_mat, annot=True, cmap='Blues', 
            xticklabels=Modules, yticklabels=Modules, fmt='g')
plt.xlabel('True Label', fontsize=15)
plt.ylabel('Prediction', fontsize=15)
file_name='hm_jetscape_ml_model_confision_matrix.png'
file_path=simulation_directory_path+file_name
plt.savefig(file_path)
plt.show()
plt.close()

classification_report_str= classification_report(y_test,y_pred)

print (classification_report_str)
file_name="hm_jetscape_ml_model_evaluation.txt"
file_path=simulation_directory_path+file_name
evaluation_file = open(file_path, "a")
evaluation_file.write(classification_report_str)
evaluation_file.close()

