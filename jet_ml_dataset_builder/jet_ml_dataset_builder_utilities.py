# Commonly used modules
import os
import sys
import subprocess
import numpy as np
import time
from time import time


# Images, plots, display, and visualization
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd


def install(package):
    import importlib.util
    # For illustrative purposes.
    spec = importlib.util.find_spec(package)
    if spec is None:
        print(package +" is not installed")
        print("Installing "+package) 
        subprocess.check_call([sys.executable,"-m" ,"pip", "install", package])
        print("Installed "+package+"\n")
        
  

# #reading/writing into files
# # !pip3 install pickle5
# install("pickle5")
# import pickle5 as pickle
install("pickle")
import pickle
def save_dataset(file_name,dataset):
    with open(file_name, 'wb') as dataset_file:
        pickle.dump(dataset,dataset_file, protocol=pickle.HIGHEST_PROTOCOL)

        
def load_dataset(file_name,is_array=False,has_validation=False, has_test=True):
    try:
        with open(file_name, 'rb') as dataset_file:
            loaded_data = pickle.load(dataset_file, encoding='latin1')
            if loaded_data is not None:
                if has_test==True:
                    if has_validation==False:
                        if is_array==False:
                            (dataset_x_train, dataset_y_train),(dataset_x_test, dataset_y_test) = loaded_data
                        else:
                            dataset_x_train = loaded_data['x_train']
                            dataset_x_test= loaded_data['x_test']
                            dataset_y_train= loaded_data['y_train']
                            dataset_y_test= loaded_data['y_test']
                        del loaded_data
                        print("dataset.x_train:",type(dataset_x_train), dataset_x_train.size, dataset_x_train.shape)
                        print("dataset.y_train:",type(dataset_y_train), dataset_y_train.size,dataset_y_train.shape)

                        print("dataset.x_test:",type(dataset_x_test), dataset_x_test.size, dataset_x_test.shape)
                        print("dataset.y_test:",type(dataset_y_test), dataset_y_test.size, dataset_y_test.shape)
                        return ((dataset_x_train, dataset_y_train), (dataset_x_test, dataset_y_test))
                    else:
                        ((dataset_x_train,dataset_y_train),(dataset_x_val,dataset_y_val),(dataset_x_test,dataset_y_test)) = loaded_data
                        del loaded_data
                        print("dataset.x_train:",type(dataset_x_train), dataset_x_train.size, dataset_x_train.shape)
                        print("dataset.y_train:",type(dataset_y_train), dataset_y_train.size,dataset_y_train.shape)

                        print("dataset.x_val:",type(dataset_x_val), dataset_x_val.size, dataset_x_val.shape)
                        print("dataset.y_val:",type(dataset_y_val), dataset_y_val.size,dataset_y_val.shape)

                        print("dataset.x_test:",type(dataset_x_test), dataset_x_test.size, dataset_x_test.shape)
                        print("dataset.y_test:",type(dataset_y_test), dataset_y_test.size, dataset_y_test.shape)

                        return ((dataset_x_train,dataset_y_train),(dataset_x_val,dataset_y_val),(dataset_x_test,dataset_y_test))
                else:
                    (dataset_x, dataset_y) = loaded_data
                    print("dataset.x:",type(dataset_x), dataset_x.size, dataset_x.shape)
                    print("dataset.y:",type(dataset_y), dataset_y.size,dataset_y.shape)
                    del loaded_data
                    return (dataset_x, dataset_y)
            else:
                print("Error: Loaded data is None.")
            # dataset=((x_train, y_train), (x_test, y_test))
            
    except pickle.UnpicklingError as e:
        print("Error while loading the pickle file:", e)
    

def store_into_dataset_file(configuration_number,y_class_label_item,data_size,simulation_directory_path,alpha_s,q0,dataset):
    file_name="config-0"+str(configuration_number)+"-"+y_class_label_item+"-alpha_s-"+str(alpha_s)+"-q0-"+str(q0)+"-simulationsize"+str(data_size)+"-dataset.pkl"
    file_name=simulation_directory_path+file_name
    print('\n########################################################################')
    print('Saving Constructed Benchmark Dataset as a file')
    save_dataset(file_name,dataset)
    print('\n########################################################################')

def save_event_items_chunk(file_name,event_items_chunks_item):
    with open(file_name, 'wb') as dataset_file:
        pickle.dump(event_items_chunks_item,dataset_file, protocol=pickle.HIGHEST_PROTOCOL)

        
def load_event_items_chunk(file_name):
    with open(file_name, 'rb') as dataset_file:
        event_items = pickle.load(dataset_file, encoding='latin1')
        return event_items
    
def load_dataset_by_y_class_label(configuration_number,data_size,simulation_directory_path,y_class_label_item,alpha_s=None,q0=None):
    print('\n########################################################################')
    print('Loading separate datasets')
    if ( alpha_s is None  or q0 is None):
        file_name="config-0"+str(configuration_number)+"-"+y_class_label_item+"-simulationsize"+str(data_size)+"-dataset.pkl"
    else:
        file_name="config-0"+str(configuration_number)+"-"+y_class_label_item+"-alpha_s-"+str(alpha_s)+"-q0-"+str(q0)+"-simulationsize"+str(data_size)+"-dataset.pkl"
    file_name=simulation_directory_path+file_name
    print("Loading Dataset from",file_name)
    ((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))= load_dataset(file_name)
    print("dataset.x_train:",type(dataset_x_train), dataset_x_train.size, dataset_x_train.shape)
    print("dataset.x_test:",type(dataset_x_test), dataset_x_test.size, dataset_x_test.shape)
    print("dataset.y_train:",type(dataset_y_train), dataset_y_train.size,dataset_y_train.shape)
    print("dataset.y_test:",type(dataset_y_test), dataset_y_test.size, dataset_y_test.shape)
    print("Sample dataset.y_test")
    print(dataset_y_test[1:10])
    dataset= ((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))
    print('\n########################################################################')
    return dataset

def get_data_splitting_index(slice_total,slice_train,data_size):
    slice_test=slice_total-slice_train
    data_splitting_index=int(data_size*(slice_train/slice_total))
    return data_splitting_index
  
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

def construct_proportional_y_column(slice_total,slice_train,column_values, data_size):
    print('\n########################################################################')
    start = time()
    print('Building Y side of the dataset')
    (y_train,y_test)=build_y_train_test_data_by_proportion(slice_total,slice_train,column_values, data_size)
    print("y_train:",type(y_train), y_train.size, y_train.shape)
    print("y_test:",type(y_test), y_test.size, y_test.shape)
    print('\n########################################################################')
    elapsed = time() - start
    print('Proportionalizing y Dataset Elapsed %.3f seconds.' % elapsed)
    print('\n########################################################################')
    return (y_train,y_test)

def concatenate_y_columns_into_dataset(dataset,alpha_s,q0):
    ((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))=dataset
    (y_train_alpha_s,y_test_alpha_s)=alpha_s
    (y_train_q0,y_test_q0)=q0
    y_train=np.array([dataset_y_train,y_train_alpha_s,y_train_q0])
    print("y_train:",type(y_train), y_train.size, y_train.shape)

    y_test=np.array([dataset_y_test,y_test_alpha_s,y_test_q0])
    print("y_test:",type(y_test), y_test.size, y_test.shape)
    return ((dataset_x_train,y_train),(dataset_x_test,y_test))

def add_alpha_s_and_q0_to_dataset(dataset,alpha_s,q0):

    import numpy as np
    print ("alpha_s:",alpha_s)
    print ("q0:",q0)
    # Assuming you have already loaded your dataset into the following variables:
    # dataset.x_train, dataset.x_test, dataset.y_train, dataset.y_test
    ((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))=dataset
    # Create a new column with a value of alpha_s input value
    alpha_s_train_column = np.full(dataset_y_train.shape, alpha_s)
    alpha_s_test_column = np.full(dataset_y_test.shape, alpha_s)

    # Create a new column with a value of q0 input value
    q0_train_column = np.full(dataset_y_train.shape, q0)
    q0_test_column = np.full(dataset_y_test.shape, q0)

    # Concatenate the new column with the target variable
    y_train_with_columns = np.column_stack((dataset_y_train, alpha_s_train_column,q0_train_column))
    y_test_with_columns = np.column_stack((dataset_y_test, alpha_s_test_column,q0_test_column))

    # Print the updated shapes and values of the resulting two columns
    print("Updated y_train shape:", y_train_with_columns.shape)
    print("Updated y_test shape:", y_test_with_columns.shape)
    print("Updated y_train values:\n", y_train_with_columns[1:10])
    print("Updated y_test values:\n", y_test_with_columns[1:10])

    return ((dataset_x_train,y_train_with_columns),(dataset_x_test,y_test_with_columns))

import getopt
import sys

def parse_parameters():
    # Remove 1st argument from the list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "hi:d:y:o:n:c:p:a:q:"

    # Long options
    long_options = ["help", "input_file_name_hadrons", "data_size", "y_class_label_items",
                    "output_dataset_file_name=", "number_of_partition", "configuration_directory",
                    "configuration_number", "alpha_s", "q0"]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)

        # Create empty dictionaries to store the tokenized parameters
        tokenized_arguments = {}
        tokenized_values = {}

        # checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--help"):
                tokenized_arguments["help"] = True
                print("displaying Help")
            elif currentArgument in ("-i", "--input_file_name_hadrons"):
                tokenized_arguments["input_file_name_hadrons"] = True
                tokenized_values["input_file_name_hadrons"] = currentValue
                print("input_file_name_hadrons: ", currentValue)
            elif currentArgument in ("-d", "--data_size"):
                tokenized_arguments["data_size"] = True
                tokenized_values["data_size"] = int(currentValue)
                print("data_size: ", currentValue)
            elif currentArgument in ("-y", "--y_class_label_items"):
                tokenized_arguments["y_class_label_items"] = True
                tokenized_values["y_class_label_items"] = [currentValue]
                print("y_class_label_items: ", currentValue)
            elif currentArgument in ("-o", "--output_dataset_file_name"):
                tokenized_arguments["output_dataset_file_name"] = True
                tokenized_values["output_dataset_file_name"] = currentValue
                print("output_dataset_file_name: ", currentValue)
            elif currentArgument in ("-n", "--number_of_partition"):
                tokenized_arguments["number_of_partition"] = True
                tokenized_values["number_of_partition"] = int(currentValue)
                print("number_of_partition: ", currentValue)
            elif currentArgument in ("-c", "--configuration_directory"):
                tokenized_arguments["configuration_directory"] = True
                tokenized_values["configuration_directory"] = currentValue
                print("configuration_directory: ", currentValue)
            elif currentArgument in ("-p", "--configuration_number"):
                tokenized_arguments["configuration_number"] = True
                tokenized_values["configuration_number"] = int(currentValue)
                print("configuration_number: ", currentValue)
            elif currentArgument in ("-a", "--alpha_s"):
                tokenized_arguments["alpha_s"] = True
                tokenized_values["alpha_s"] = float(currentValue)
                print("alpha_s: ", currentValue)
            elif currentArgument in ("-q", "--q0"):
                tokenized_arguments["q0"] = True
                tokenized_values["q0"] = float(currentValue)
                print("q0: ", currentValue)

        return tokenized_arguments, tokenized_values

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        return {}, {}

import os
from os import path, makedirs

def set_directory_paths():
    print('\n########################################################################')
    print('Checking the running platforms and setting the directory path\n')
    dataset_directory_path = ''
    simulation_directory_path = ''
    
    import platform
    print("Python version: "+platform.python_version())
    
    running_os=platform.system()
    print("OS: "+running_os)
    print("OS version: "+platform.release())
    try:
        from google.colab import drive
        COLAB = True
    except:
        COLAB = False
    print("running on Colab: "+str(COLAB))
    if COLAB == True:
        drive.mount('/content/drive')
        dataset_directory_path = '/content/drive/MyDrive/Projects/110_JetscapeMl/hm.jetscapeml.data/'
        simulation_directory_path = dataset_directory_path + 'simulation_results/'
    elif 'Linux' in running_os:
        dataset_directory_path = '/wsu/home/gy/gy40/gy4065/hm.jetscapeml.data/'
        simulation_directory_path = dataset_directory_path + 'simulation_results/'
    else:
        # dataset_directory_path = 'G:\\My Drive\\Projects\\110_JetscapeMl\\hm.jetscapeml.data\\'
        dataset_directory_path = 'D:\\Projects\\110_JetscapeMl\\hm.jetscapeml.data\\'
        simulation_directory_path = dataset_directory_path + 'simulation_results\\'

    print('Dataset Directory Path: ' + dataset_directory_path)
    if not path.exists(simulation_directory_path):
        makedirs(simulation_directory_path)
    print('Simulation Results Path: ' + simulation_directory_path)
    print('########################################################################\n')
    return dataset_directory_path, simulation_directory_path

import numpy as np
from sklearn.utils import shuffle

def merge_and_shuffle_datasets(dataset1, dataset2):
    # Convert tuples to dictionaries

    ((dataset1_x_train,dataset1_y_train),(dataset1_x_test,dataset1_y_test))=dataset1
    ((dataset2_x_train,dataset2_y_train),(dataset2_x_test,dataset2_y_test))=dataset2
    # Merge datasets
    merged_dataset = {
        'x_train': np.concatenate((dataset1_x_train, dataset2_x_train)),
        'x_test': np.concatenate((dataset1_x_test, dataset2_x_test)),
        'y_train': np.concatenate((dataset1_y_train, dataset2_y_train)),
        'y_test': np.concatenate((dataset1_y_test, dataset2_y_test))
    }

    # Shuffle training data
    # merged_dataset['x_train'], merged_dataset['y_train'] = shuffle(merged_dataset['x_train'], merged_dataset['y_train'])
    num_samples = merged_dataset['x_train'].shape[0]
    indices = np.random.permutation(num_samples)

    print("indices",indices[1:100])
    merged_dataset['x_train'] = merged_dataset['x_train'][indices]
    merged_dataset['y_train'] = merged_dataset['y_train'][indices]

    print("merged_dataset x_train shape:", merged_dataset['x_train'].shape)
    print("merged_dataset x_test shape:", merged_dataset['x_test'].shape)
    print("merged_dataset y_train shape:", merged_dataset['y_train'].shape)
    print("merged_dataset y_test shape:", merged_dataset['y_test'].shape)
    print("merged_dataset y_train values:\n", merged_dataset['y_train'][1:100])
    print("merged_dataset y_test values:\n", merged_dataset['y_test'][1:10])
    return merged_dataset

def shuffle_dataset(dataset):
    if dataset is None:
        return dataset
    
    ((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))=dataset
   
    # Shuffle training data
    # merged_dataset['x_train'], merged_dataset['y_train'] = shuffle(merged_dataset['x_train'], merged_dataset['y_train'])
    num_samples = dataset_x_train.shape[0]
    indices = np.random.permutation(num_samples)

    print("indices",indices[1:100])
    dataset_x_train = dataset_x_train[indices]
    dataset_y_train = dataset_y_train[indices]

    print("shuffled_dataset x_train shape:", dataset_x_train.shape)
    print("shuffled_dataset x_test shape:", dataset_x_test.shape)
    print("shuffled_dataset y_train shape:",dataset_y_train.shape)
    print("shuffled_dataset y_test shape:", dataset_y_test.shape)
    print("shuffled_dataset y_train values:\n", dataset_y_train[1:100])
    print("shuffled_dataset y_test values:\n", dataset_y_test[1:10])
    dataset=((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))
    return dataset
    

def merge_datasets(dataset1, dataset2):
    # this piece of code somehow is putting the x and y side for the merged document reverse
    # Convert tuples to dictionaries

    if dataset1 is None and dataset2 is None:
        return None
    if dataset1 is None:
        return dataset2
    
    ((dataset1_x_train,dataset1_y_train),(dataset1_x_test,dataset1_y_test))=dataset1
    ((dataset2_x_train,dataset2_y_train),(dataset2_x_test,dataset2_y_test))=dataset2
    # Merge datasets
    x_train= np.concatenate((dataset1_x_train, dataset2_x_train))
    x_test=np.concatenate((dataset1_x_test, dataset2_x_test))
    y_train=np.concatenate((dataset1_y_train, dataset2_y_train))
    y_test= np.concatenate((dataset1_y_test, dataset2_y_test))
    
    print("merged_dataset x_train shape:", x_train.shape)
    print("merged_dataset x_test shape:", x_test.shape)
    print("merged_dataset y_train shape:", y_train.shape)
    print("merged_dataset y_test shape:", y_test.shape)

    merged_dataset=None
    merged_dataset=((x_train,y_train),(x_test,y_test))
    return merged_dataset

def store_merged_dataset_into_npz(merged_dataset, alpha_s, q0, total_size, y_class_label_items, configuration_number, simulation_directory_path):
    # Create file name
    class_labels = '_'.join(y_class_label_items)
    file_name = f"config_0{configuration_number}_alpha_{alpha_s}_q0_{q0}_{class_labels}_size_{total_size}_shuffled.npz"
    file_name=simulation_directory_path+file_name
    print("Save merged dataset as ",file_name)
    # save_dataset(file_name,merged_dataset)
    
    np.savez(file_name, **merged_dataset)

    print("Merged dataset saved as:", file_name)

def store_merged_dataset(merged_dataset, alpha_s, q0, total_size, y_class_label_items, configuration_number, simulation_directory_path):
    # Create file name
    class_labels = '_'.join(y_class_label_items)
    file_name = f"config_0{configuration_number}_alpha_{alpha_s}_q0_{q0}_{class_labels}_size_{total_size}_shuffled.pkl"
    file_name=simulation_directory_path+file_name
    print("Save merged dataset as ",file_name)
    save_dataset(file_name,merged_dataset)
    print("Merged dataset saved as:", file_name)
def load_dataset_from_npz(npz_file):
    # Load the dataset from the NPZ file
    data = np.load(npz_file)

    # Create the dataset dictionary
    dataset = {
        'x_train': data['x_train'],
        'x_test': data['x_test'],
        'y_train': data['y_train'],
        'y_test': data['y_test']
    }

    return dataset
def get_label_items():
    print ('Aggregatring all parameters values')
    eloss_items=['MMAT','MLBT']
    alpha_s_items=[0.2 ,0.3 ,0.4]
    q0_items=[1.5 ,2.0 ,2.5]
    data_dict = {
        "eloss_items": eloss_items,
        "alpha_s_items": alpha_s_items,
        "q0_items": q0_items
    }
    print("label_items:\n",data_dict)
    return data_dict

def get_labels_str(label_items_dict=None):
  if label_items_dict==None:
      label_items_dict = get_label_items()
      return get_labels_str(label_items_dict)
  print("Building required params for the loading the dataset file")

  data_dict = {
      "eloss_items_str":'_'.join(label_items_dict['eloss_items']),
      "alpha_s_items_str":'_'.join(map(str, label_items_dict['alpha_s_items'])),
      "q0_items_str":'_'.join(map(str, label_items_dict['q0_items'])),
  }
  print("labels_str:\n",data_dict)
  return data_dict

def generate_simulation_path(simulation_directory_path,classifying_parameter, label_str_dict, dataset_size, n_epochs, fold):
    """
    Generate a simulation path based on input parameters.

    Parameters:
    - simulation_directory_path (str): The directory path where the simulation results will be stored.
    - label_str_dict (dict): A dictionary containing label strings.
    - dataset_size (int): The size of the dataset.
    - n_epochs (int): The number of epochs for training.
    - fold (int): The fold number for cross-validation.

    Returns:
    - current_simulation_path (str): The generated simulation path.
    """

    # Print simulation directory path
    print("simulation_directory_path:", simulation_directory_path)
    
    key=classifying_parameter+"_items_str"
    classifying_parameter_label_str=label_str_dict[key]
    # Generate simulation path components
    simulation_path = f'{simulation_directory_path}jetml_pointnet_classification_{classifying_parameter}_{classifying_parameter_label_str}'
    print("simulation_path:", simulation_path)

    current_simulation_name = f'_size_{dataset_size}'
    current_simulation_path = simulation_path + current_simulation_name

    current_simulation_name = f'_epochs_{n_epochs}'
    current_simulation_path = current_simulation_path + current_simulation_name

    current_simulation_name = f'_fold_{fold}'
    current_simulation_path = current_simulation_path + current_simulation_name

    return current_simulation_path

def scale_dataset_images(dataset_x):
    """
    Scale each image in the dataset_x between 0 and 1 using Min-Max scaling.

    Parameters:
    - dataset_x (numpy.ndarray): The dataset containing images.

    Returns:
    - scaled_dataset_x (numpy.ndarray): The scaled dataset.
    """
    # Calculate the minimum and maximum values for each image
    min_vals = np.min(dataset_x, axis=(1, 2), keepdims=True)
    max_vals = np.max(dataset_x, axis=(1, 2), keepdims=True)

    # Scale each image between 0 and 1
    scaled_dataset_x = (dataset_x - min_vals) / (max_vals - min_vals)

    return scaled_dataset_x

def get_dataset(size: int, label_str_dict: dict, dataset_directory_path: str, working_column: int = 0,scale_x=True):
    """
    Loads a dataset of specified size and extracts the specified column for classification.

    Parameters:
    - size (int): The size of the dataset. It should be an integer representing the size of the dataset. 
                  Valid sizes are 1000, 10000, 100000, or 1000000.
    - label_str_dict (dict): A dictionary containing string labels for various parameters used in the dataset file name construction.
    - dataset_directory_path (str): The directory path where the dataset files are located.
    - working_column (int, optional): The index of the column to be extracted for classification. Default is 0.

    Returns:
    - dataset_x (numpy.ndarray): The features of the dataset.
    - dataset_y (numpy.ndarray): The labels corresponding to the features.

    Example:
    ```python
    dataset_x, dataset_y = get_dataset(1000, label_str_dict, "/path/to/dataset_directory/", working_column=1)
    ```
    """

    dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{label_str_dict['alpha_s_items_str']}_q0_{label_str_dict['q0_items_str']}_{label_str_dict['eloss_items_str']}_size_{size}_shuffled.pkl"

    dataset_file_name = dataset_directory_path + dataset_file_name

    print("Loading the whole dataset")
    dataset = load_dataset(dataset_file_name, has_test=False)
    (dataset_x, dataset_y) = dataset
    
    if(scale_x==True):
        print("Scaling the datset_x each image between 0 and 1")
        dataset_x = scale_dataset_images(dataset_x)
    print(f'Extract the working column#{working_column} for classification')
    dataset_y = dataset_y[:, working_column]
    print("dataset.x:",type(dataset_x), dataset_x.size, dataset_x.shape)
    print("dataset.y:",type(dataset_y), dataset_y.size,dataset_y.shape)
    print("dataset.y(working_column) sample",dataset_y[:10])

    return dataset_x, dataset_y




