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


print("Python version: "+platform.python_version())
print("Tensorflow version: "+tf.__version__)

dataset_directory_path=''
simulation_directory_path=''

if COLAB == True:
  drive.mount('/content/drive')
  dataset_directory_path='/content/drive/MyDrive/Projects/110_JetscapeMl/hm.jetscapeml.data/simulation_results/'
  simulation_directory_path=dataset_directory_path+'simulation-results-deep-model-cnn-01-1200K-config-03-epoch-30/'
elif 'Linux' in running_os:
  dataset_directory_path='/wsu/home/gy/gy40/gy4065/hm.jetscapeml.data/simulation_results/'
  simulation_directory_path=dataset_directory_path+'simulation-results-deep-model-cnn-01-1200K-config-03-epoch-30/'
else:
  dataset_directory_path= 'G:\\My Drive\\Projects\\110_JetscapeMl\\hm.jetscapeml.data\\simulation_results\\'
  simulation_directory_path=dataset_directory_path+'simulation-results-deep-model-cnn-01-1200K-config-03-epoch-30\\'
print('Dataset Directory Path: '+dataset_directory_path)

# dataset_file_name='jetscape-ml-benchmark-dataset-2k-randomized.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-2k-shuffled.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-200k-shuffled-01.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-1200k-momentum-shuffled.pkl'
# dataset_file_name='config-01-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-02-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
dataset_file_name='config-03-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-04-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-05-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
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


from os import path, makedirs
from glob import glob
import random
import numpy as np
import tensorflow as tf
import json
import shutil
from time import time

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

seed = np.random.randint(1,100)
print('Seed for random numbers: {}'.format(seed))
print('Tensorflow version: {}'.format(tf.__version__))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


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



def calculate_dataset_x_amax_value(x_dataset):
  x_train=x_dataset[0]
  x_test=x_dataset[1]
  max_x=np.amax([np.amax(x_train), np.amax(x_test)])
  return max_x

def calculate_dataset_x_max_value(x_dataset):
  x_train=x_dataset[0]
  x_test=x_dataset[1]
  max_x=np.max([np.max(x_train), np.max(x_test)])
  return max_x

def calculate_dataset_x_norm_value(x_dataset):
  x_train=x_dataset[0]
  x_test=x_dataset[1]
  x_train_norm = np.linalg.norm(x_train)       # To find the norm of the array
  print("x_train_norm",x_train_norm)
  x_test_norm = np.linalg.norm(x_test)       # To find the norm of the array
  print("x_test_norm",x_test_norm)
  max_x=np.max([np.max(x_train_norm), np.max(x_test_norm)])
  return max_x
  
  # max_x=np.max([np.max(x_train), np.max(x_test)])
  # return max_x

def normalize_dataset_x_value_range_between_0_and_1(x_dataset,max_x):
  x_train=x_dataset[0]
  x_test=x_dataset[1]

  # Normalize the data to a 0.0 to 1.0 scale for faster processing
  x_train, x_test = x_train / max_x, x_test / max_x
  return (x_train, x_test)


#Normalizing Phase
x_dataset=(x_train,x_test)
# norm_max_x=calculate_dataset_x_norm_value(x_dataset)
# amax_x=calculate_dataset_x_amax_value(x_dataset)
# max_x=calculate_dataset_x_max_value(x_dataset)
# x_train,x_test=normalize_dataset_x_value_range_between_0_and_1(x_dataset,norm_max_x)

# image_frame_size=32

print("\n#############################################################")

print("Normalizing Dataset X: maximum sum of transfer momentum in the dataset: ")
# print("norm_max_x",norm_max_x)
# print("amax",amax_x)
# print("max_x",max_x)

print("#############################################################\n")

# exit()


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


## create a directory to save the best model

save_dir = (simulation_directory_path+'models_{}_vs_{}_{}_ch{}').format(Modules[0], Modules[1], kind, len(observables))
if not path.exists(save_dir):
    makedirs(save_dir)
print('Directory to save models: {}'.format(save_dir))


# In[ ]:


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Adagrad


def get_callbacks(monitor, save_dir):
    mode = None
    if 'loss' in monitor:
        mode = 'min'
    elif 'accuracy' in monitor:
        mode = 'max'
    assert mode != None, 'Check the monitor parameter!'

    es = EarlyStopping(monitor=monitor, mode=mode, patience=10,
                      min_delta=0., verbose=1)
    rlp = ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.2, patience=5,
                            min_lr=0.001, verbose=1)
    mcp = ModelCheckpoint(path.join(save_dir, 'hm_jetscape_ml_model_best.h5'), monitor=monitor, 
                          save_best_only=True, mode=mode, verbose=1)
    
    return [es, rlp, mcp]

def conv2d_layer_block(prev_layer, filters, dropout_rate, input_shape=None):
    if input_shape != None:
        prev_layer.add(Conv2D(filters=filters, kernel_size=5,
                              kernel_initializer='he_uniform',
                              padding='same',
                              activation='relu',
                              kernel_regularizer=l2(l=0.02),
                              input_shape=input_shape
                             )
                      )
    else:
        prev_layer.add(Conv2D(filters=filters, kernel_size=5,
                              kernel_initializer='he_uniform',
                              padding='same',
                              activation='relu',
                              kernel_regularizer=l2(l=0.02)
                             )
                      )
    prev_layer.add(Conv2D(filters=filters, kernel_size=5,
                              kernel_initializer='he_uniform',
                              padding='same',
                              activation='relu',
                              kernel_regularizer=l2(l=0.02)
                             )
                      )    
    prev_layer.add(MaxPool2D(pool_size=(2, 2)))
    prev_layer.add(Dropout(dropout_rate))
    
    return prev_layer

def fc_layer_block(prev_layer, units, dropout_rate, last_layer=False):
    if last_layer == False:
        prev_layer.add(Dense(units, activation='relu',
                             kernel_initializer='he_uniform',
                             kernel_regularizer=l2(l=0.02)
                            )
                      )
        prev_layer.add(Dropout(dropout_rate))
    else:
        prev_layer.add(Dense(1, activation='sigmoid'))

    return prev_layer

def CNN_model(input_shape, lr, dropout1, dropout2):
    model = Sequential()
    # model.add(Rescaling(1./max_x))
    model = conv2d_layer_block(model, 256, dropout1, input_shape)
    model = conv2d_layer_block(model, 256, dropout1)
    model = conv2d_layer_block(model, 256, dropout1)
    model = conv2d_layer_block(model, 256, dropout1)
    #model = conv2d_layer_block(model, 128, dropout1)
    model.add(Flatten())
    model = fc_layer_block(model, 1024, dropout2)
    model = fc_layer_block(model, 1024, dropout2)
    model = fc_layer_block(model, 1024, dropout2)
    model = fc_layer_block(model, 1024, dropout2)
    model = fc_layer_block(model, 1, None, last_layer=True)
    
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


# In[ ]:


## parameers for training
n_epochs = 30
batch_size = 256
input_shape = x_train_reshaped.shape[1:]
monitor='val_accuracy' #'val_accuracy' or 'val_loss'
lr = 5e-6
dropout1, dropout2 = 0.2, 0.2


# In[ ]:


def train_network(train_set, val_set, n_epochs, lr, batch_size, monitor):
    tf.keras.backend.clear_session()
    X_train = train_set[0]
    Y_train = train_set[1]
    model = CNN_model(input_shape, lr, dropout1, dropout2)
    callbacks = get_callbacks(monitor, save_dir)
    
    model.summary()
    
    start = time()
    history = model.fit(X_train, Y_train, epochs=n_epochs, verbose=1, batch_size=batch_size, 
                        validation_data=val_set, shuffle=True, callbacks=callbacks)

    train_time = (time()-start)/60.
    return history, train_time


# In[ ]:


# training and validation sets
train_set, val_set = (x_train_reshaped, y_train), (x_val_reshaped, y_val)

# train the network
history, train_time = train_network(train_set, val_set, n_epochs, lr, batch_size, monitor)
file_name='hm_jetscape_ml_model_history.csv'
file_path=simulation_directory_path+file_name
pd.DataFrame.from_dict(history.history).to_csv(file_path,index=False)


file_name='hm_jetscape_ml_model_history.npy'
file_path=simulation_directory_path+file_name
np.save(file_path,history.history)


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
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy History')
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

