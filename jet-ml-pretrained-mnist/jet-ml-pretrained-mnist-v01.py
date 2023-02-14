#!/usr/bin/env python
# coding: utf-8

# ## Part 0: Prerequisites:
# 
# We recommend that you run this this notebook in the cloud on Google Colab (see link with icon at the top) if you're not already doing so. It's the simplest way to get started. You can also [install TensorFlow locally](https://www.tensorflow.org/install/).
# 
# Note that there's [tf.keras](https://www.tensorflow.org/guide/keras) (comes with TensorFlow) and there's [Keras](https://keras.io/) (standalone). You should be using [tf.keras](https://www.tensorflow.org/guide/keras) because (1) it comes with TensorFlow so you don't need to install anything extra and (2) it comes with powerful TensorFlow-specific features.

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
  simulation_directory_path=dataset_directory_path+'simulation-results-nn-mnist-1200K-config-08/'
elif 'Linux' in running_os:
  dataset_directory_path='/wsu/home/gy/gy40/gy4065/hm.jetscapeml.data/simulation_results/'
  simulation_directory_path=dataset_directory_path+'simulation-results-nn-mnist-1200K-config-08/'
else:
  dataset_directory_path= 'G:\\My Drive\\Projects\\110_JetscapeMl\\hm.jetscapeml.data\\simulation_results\\'
  simulation_directory_path=dataset_directory_path+'simulation-results-nn-mnist-1200K-config-08\\'
print('Dataset Directory Path: '+dataset_directory_path)

#dataset_file_name='jetscape-ml-benchmark-dataset-2k-randomized.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-2000.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-200k-shuffled-01.pkl'
# dataset_file_name='jetscape-ml-benchmark-dataset-matter-vs-lbt-1200k-momentum-shuffled.pkl'
# dataset_file_name='config-01-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-02-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-03-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-04-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-05-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
dataset_file_name='config-08-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-07-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-08-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
# dataset_file_name='config-09-matter-vs-lbt-simulationsize1200000-dataset-momentum-shuffled.pkl'
print("Dataset file name: "+dataset_file_name)

if not path.exists(simulation_directory_path):
    makedirs(simulation_directory_path)
print('Simulation Results Path: '+simulation_directory_path)
print('########################################################################\n')
print('\nLoading/Installing Package => End\n\n')


# ## 1. Load Data into a Numpy Array  
# Downloading the data file onto my desktop/server and loading it locally.  
# 
# (x_train, y_train), (x_test, y_test) = jetscapeMl.load_data()  
# 
# **After the load:**   
# x_train contains an array of 32x32.  
# The y_train vector contains the corresponding labels for these.  
# x_test contains an arrays of 32x32.  
# The y_test vector contains the corresponding labels for these.

# ##Saving and Loading Dataset Methods Implementation

# In[ ]:


def save_dataset(file_name,dataset):
    with open(file_name, 'wb') as dataset_file:
        pickle.dump(dataset,dataset_file, protocol=pickle.HIGHEST_PROTOCOL)

        
def load_dataset(file_name):
    with open(file_name, 'rb') as dataset_file:
        (x_train, y_train), (x_test, y_test) = pickle.load(dataset_file, encoding='latin1')
        dataset=((x_train, y_train), (x_test, y_test))
        return dataset


# ## 2. Use Matplotlib to visualize one record.  
# I set the colormap to Grey and ColorMap. There are a bunch of other colormap choices if you like bright visualizations. Try magma or any of the other  choice in the [docs](https://matplotlib.org/tutorials/colors/colormaps.html).

# In[ ]:


def plot_event(image_frame_size,event_matrix,file_name):
  plt.imshow(event_matrix.reshape(image_frame_size, image_frame_size), cmap=cm.Greys)
  cb = plt.colorbar()
  cb.set_label("Hit Frequency")
  
  file_path=simulation_directory_path+file_name
  plt.savefig(file_path)
# # Funtionality Testing
# # Plotting sample Random Event Histogram on gray scale
# image_frame_size=32
# plot_event(image_frame_size,counts,file_name='sample_random_event_histogram_32x32_grayscale.png')


# #Loading Dataset
# **First learning step**

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


# ## 3. Plot a bunch of records to see sample data  
# Basically, use the same Matplotlib commands above in a for loop to show 20 records from the train set in a subplot figure. We also make the figsize a bit bigger and remove the tick marks for readability.
# ** TODO: try to make the subplot like the below from the first project meeting

# In[ ]:


def plot_20_sample_events(events_matrix_items):
  # images = x_train[0:18]
  # fig, axes = plt.subplots(3, 6, figsize=[9,5])
  images = events_matrix_items
  fig, axes = plt.subplots(2, 10, figsize=[15,5])

  for i, ax in enumerate(axes.flat):
      current_plot= ax.imshow(x_train[i].reshape(32, 32), cmap=cm.Greys)
      ax.set_xticks([])
      ax.set_yticks([])     
  

  file_name='hm_jetscape_ml_plot_20_sample_events.png'
  file_path=simulation_directory_path+file_name
  plt.savefig(file_path)

  plt.show()
  plt.close()
#Plotting 20 Sample Events Phase
events_matrix_items=[x_train[0:10],x_train[1500:10]]
plot_20_sample_events(events_matrix_items)


# ## 4. Show distribution of training data labels   
# The training data is about evenly distributed across all nine digits. 

# In[ ]:


def plot_y_train_dataset_distribution(y_train):
  unique_class_labels,positions = np.unique(y_train,return_inverse=True)


  counts = np.bincount(positions)
  plt.bar(unique_class_labels, counts)
  plt.title("Dataset classification vector's distribution")
  
  
  file_name='hm_jetscape_ml_plot_y_train_dataset_distribution.png'
  file_path=simulation_directory_path+file_name
  plt.savefig(file_path)

  plt.show()
  plt.close()
  print("\n#############################################################")
  print("Classification vector statistics:")
  print(unique_class_labels)
  unique_class_labels,positions = np.unique(y_train,return_inverse=True)
  print(counts)
  print(unique_class_labels)
  print("Sample 20 head labels:")
  print(positions[:20])
  print("#############################################################\n")

#Checking Train Dataset Y Distribution
plot_y_train_dataset_distribution(y_train)


# # Changing classification labels from Literal to Numeric
# 
# ---
# 

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


# ## Normalizing the Dataset X
# For training the model, the dataset needs to be normalized, meaning all of the dataset values should me between zero and one. This can be done by finding the maximum values over the dataset X side values and devide all the element by it.

# In[ ]:


def calculate_dataset_x_max_value(x_dataset):
  x_train=x_dataset[0]
  x_test=x_dataset[1]
  max_x=np.amax([np.amax(x_train), np.amax(x_test)])
  return max_x

def normalize_dataset_x_value_range_between_0_and_1(x_dataset,max_x):
  x_train=x_dataset[0]
  x_test=x_dataset[1]

  # Normalize the data to a 0.0 to 1.0 scale for faster processing
  x_train, x_test = x_train / max_x, x_test / max_x
  return (x_train, x_test)


#Normalizing Phase
x_dataset=(x_train,x_test)
max_x=calculate_dataset_x_max_value(x_dataset)
x_train,x_test=normalize_dataset_x_value_range_between_0_and_1(x_dataset,max_x)

image_frame_size=32

print("\n#############################################################")

print("Normalizing Dataset X: maximum sum of transfer momentum in the dataset: ")
print(max_x)

print("#############################################################\n")


# ## Defining Validation Dataset from Train Dataset

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


# ## 5.1 Apply Keras/TensorFlow neural network  
# Use tensorflow to train the model with 1600/180K training records, compile the model, and classify 400/20K test records and calcualte the accuracy accuracy.  
# **Create the model**  
# Build the keras model by stacking layers into the network. Our model here has four layers:
# - Flatten reshapes the data into a 1-dimensional array
# - [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) tells the model to use output arrays of shape (*, 512) and sets rectified linear [activation function](https://keras.io/activations/). 
# - [Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) applies dropout to the input to help avoid overfitting.
# - The next Dense line condenses the ouput into probabilities for each of the 2 Energy Loss Classes.
# 
# **Compile the model**   
# - [Adam](https://keras.io/optimizers/) is an optimization algorithm that uses stochastic gradient descent to update network weights.
# - Sparse categorical crossentropy is a [loss function](https://keras.io/losses/) that is required to compile the model. The loss function measures how accurate the model is during training. We want to minimize this function to steer the model in the right direction.
# - A metric is a function that is used to judge the performance of your model. We're using accuracy of our predictions as compared to y_test as our metric.  
# Lastly, we fit our training data into the model, with several training repetitions (epochs), then evaluate our test data. 
# 
# Our final result is about ?% accuracy in classifying 400/20K events in the test set. You can try tweaking this model with different settings to get a better score. An easy tweak is increasing the epochs, which improves accuracy at the expense of time. Follow the links to the Keras layer docs above and try different options for Dense output, activation functions, optimization algorithms and loss functions.
# 
# ### Train the model
# 
# Training the neural network model requires the following steps:
# 
# 1. Feed the training data to the model—in this example, the `x_train` and `y_train` arrays, which are the images and labels.
# 2. The model learns to associate images and labels.
# 3. We ask the model to make predictions about a test set—in this example, the `x_test` array. We verify that the predictions match the labels from the `y_test` array. 
# 
# To start training,  call the `model.fit` method—the model is "fit" to the training data:
# *Setting number of Epochs to 1000*

# In[ ]:


# this helps makes our output less verbose but still shows progress
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def build_and_train_model():
  # model = tf.keras.models.Sequential([
  #   tf.keras.layers.Flatten(input_shape=(image_frame_size, image_frame_size)),
  #   tf.keras.layers.Dense(max_x+1, activation=tf.nn.relu),
  #   tf.keras.layers.Dropout(0.25),
  #   tf.keras.layers.Dense(2, activation=tf.nn.softmax)
  # ])
  model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(image_frame_size, image_frame_size)),
  tf.keras.layers.Dense(max_x+1, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])
  model.summary()

  model.compile(optimizer=tf.optimizers.Adam(), 
                loss='binary_crossentropy',
                metrics=['acc'])
  print("Fit model on training data")
  # fit(object, x = NULL, y = NULL, batch_size = NULL, epochs = 10,
  #   verbose = getOption("keras.fit_verbose", default = 1),
  #   callbacks = NULL, view_metrics = getOption("keras.view_metrics",
  #   default = "auto"), validation_split = 0, validation_data = NULL,
  #   shuffle = TRUE, class_weight = NULL, sample_weight = NULL,
  #   initial_epoch = 0, steps_per_epoch = NULL, validation_steps = NULL,
  #   ...)
  # -> object : the model to train.      
  # -> X : our training data. Can be Vector, array or matrix      
  # -> Y : our training labels. Can be Vector, array or matrix       
  # -> Batch_size : it can take any integer value or NULL and by default, it will
  # be set to 32. It specifies no. of samples per gradient.      
  # -> Epochs : an integer and number of epochs we want to train our model for.      
  # -> Verbose : specifies verbosity mode(0 = silent, 1= progress bar, 2 = one
  # line per epoch).      
  # -> Shuffle : whether we want to shuffle our training data before each epoch.      
  # -> steps_per_epoch : it specifies the total number of steps taken before
  # one epoch has finished and started the next epoch. By default it values is set to NULL.

  start_time =time.perf_counter()

  no_epoch=2
  history = model.fit(
      x_train,
      y_train,
      batch_size=64,
      epochs=no_epoch,
      # We pass some validation for
      # monitoring validation loss and metrics
      # at the end of each epoch
      validation_data=(x_val, y_val))

  file_name='trained_model.h5'
  file_path=simulation_directory_path+file_name
  model.save(file_path)
  acc_train = history.history['acc']
  acc_val = history.history['val_acc']
  print(acc_train)
  print(acc_train)
  epochs = range(1,no_epoch+1)
  plt.plot(epochs,acc_train, 'g', label='training accuracy')
  plt.plot(epochs, acc_val, 'b', label= 'validation accuracy')
  plt.title('Training and Validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()
  plt.close()
  elapsed_time=time.perf_counter() - start_time
  print('Elapsed %.3f seconds.' % elapsed_time)

# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
# , verbose=0, validation_split = 0.1,
#                     callbacks=[early_stop, PrintDot()]


# ## 5.2 Apply Keras/TensorFlow with MNIST pretrained model
# 

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




def CNN_model(input_shape, lr, dropout):
    model = Sequential()
    
    model.add(Flatten(input_shape=(image_frame_size, image_frame_size)))
    model.add(Dense(max_x+1, activation=tf.nn.relu))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=tf.nn.sigmoid))
    
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


# In[ ]:


# print("\n#############################################################")
# print("Reshaping dataset X-side dimension to be fit in the defined convolutional :")

# print("\nX train:")
# print(x_train.shape)
# print (x_train.shape[0],x_train.shape[1],x_train.shape[2])
# x_train_reshaped=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# print(x_train_reshaped.shape)

# print("\nX val:")
# print(x_val.shape)
# print (x_val.shape[0],x_val.shape[1],x_val.shape[2])
# x_val_reshaped=x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)
# print(x_train_reshaped.shape)

# print("\nX test:")
# print(x_test.shape)
# print (x_test.shape[0],x_test.shape[1],x_test.shape[2])
# x_test_reshaped=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
# print(x_test_reshaped.shape)
# print("#############################################################\n")


# In[ ]:


## parameers for training
n_epochs = 50
# n_epochs=2
batch_size = 256
# input_shape = x_train_reshaped.shape[1:]
input_shape = x_train.shape[1:]
monitor='val_accuracy' #'val_accuracy' or 'val_loss'
lr = 5e-6
dropout= 0.25


# In[ ]:


from time import time
def train_network(train_set, val_set, n_epochs, lr, batch_size, monitor):
    tf.keras.backend.clear_session()
    X_train = train_set[0]
    Y_train = train_set[1]
    model = CNN_model(input_shape, lr, dropout)
    callbacks = get_callbacks(monitor, save_dir)
    
    model.summary()
    
    start = time()
    history = model.fit(X_train, Y_train, epochs=n_epochs, verbose=1, batch_size=batch_size, 
                        validation_data=val_set, shuffle=True, callbacks=callbacks)
    train_time = (time()-start)/60.
    return history, train_time


# In[ ]:


# training and validation sets
# train_set, val_set = (x_train_reshaped, y_train), (x_val_reshaped, y_val)
train_set, val_set = (x_train, y_train), (x_val, y_val)

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

    plt.figure(figsize=(8, 2.5), dpi=100)

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
_, train_acc = best_model.evaluate(x_train, y_train, verbose=0)
_, val_acc = best_model.evaluate(x_val, y_val, verbose=0)
_, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    
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
y_pred = best_model.predict_classes(x_test)

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

