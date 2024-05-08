#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(1,'/wsu/home/gy/gy40/gy4065/hm.jetscapeml.source')

import numpy as np


# In[ ]:


print('Loading/Installing Package => Begin\n\n')

import jet_ml_dataset_builder.jet_ml_dataset_builder_utilities as util

print('\n########################################################################')
print('Checking the running platforms\n')

from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import set_directory_paths
# Call the function and retrieve the dataset_directory_path and simulation_directory_path
dataset_directory_path, simulation_directory_path = set_directory_paths()

# Access the dataset_directory_path and simulation_directory_path
print("Dataset Directory Path:", dataset_directory_path)
print("Simulation Directory Path:", simulation_directory_path)
print('########################################################################\n')


print('\nLoading/Installing Package => End\n\n')


# In[ ]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import  parse_parameters

# Call the function and retrieve the tokenized parameters
tokenized_arguments, tokenized_values = parse_parameters()

# Access the tokenized arguments and values
print("Tokenized Arguments:")
for argument in tokenized_arguments:
    print(argument)

print("\nTokenized Values:")
for argument, value in tokenized_values.items():
    print(f"{argument}: {value}")

y_class_label_items=['MMAT','MLBT']
alpha_s_items=[0.2 ,0.3 ,0.4]
q0_items=[1.5 ,2.0 ,2.5]

print("y_class_label_items:",y_class_label_items)
print("alpha_s_items:",alpha_s_items)
print("q0_items:",q0_items)


# # Loading the dataset
# 

# In[ ]:


print("Building required params for the loading the dataset file")

class_labels_str = '_'.join(y_class_label_items)
alpha_s_items_str='_'.join(map(str, alpha_s_items))
q0_items_str='_'.join(map(str, q0_items))
total_size=9*1200000
# for shuffled_y_processed
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_split_train_datasets/train_split_0.pkl"
# for shuffled
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled.pkl"
dataset_file_name=simulation_directory_path+dataset_file_name
print("dataset_file_name:",dataset_file_name)


# In[ ]:


# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import load_dataset
# dataset=load_dataset (dataset_file_name,has_test=False)
# (dataset_x, dataset_y)=dataset
# print("dataset y_train values:\n", dataset_x[1:10])
# print("dataset y_test values:\n", dataset_y[1:10])


# In[ ]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import load_dataset
dataset=load_dataset (dataset_file_name)
((dataset_x, dataset_y),(dataset_x_test,dataset_y_test))=dataset
print("After loading the dataset")
print("dataset.x_train:",type(dataset_x), dataset_x.size, dataset_x.shape)
print("dataset.y_train:",type(dataset_y), dataset_y.size,dataset_y.shape)

print("dataset.x_test:",type(dataset_x_test), dataset_x_test.size, dataset_x_test.shape)
print("dataset.y_test:",type(dataset_y_test), dataset_y_test.size, dataset_y_test.shape)


# In[ ]:


# def split_dataset(dataset_x, dataset_y, dataset_size):
#     """
#     Split a dataset into proportioned x and y.

#     Parameters:
#     - dataset: Dictionary or object containing 'x' and 'y' numpy arrays.
#     - dataset_size: Number of samples for the final dataset.


#     Returns:
#     - x, y: Numpy arrays for the dataset.
#     """
#     # Create the training dataset
#     x = dataset_x[:dataset_size]
#     y = dataset_y[:dataset_size]

#     return (x, y)


# In[ ]:


def split_dataset(dataset_x, dataset_y, train_size, test_size = None):
    """
    Split a dataset into training and test sets.

    Parameters:
    - dataset: Dictionary or object containing 'x' and 'y' numpy arrays.
    - train_size: Number of samples for the training set.
    - test_size: Number of samples for the test set.

    Returns:
    - x_train, y_train, x_test, y_test: Numpy arrays for the training and test sets.
    """
    
    x_train = dataset_x[:train_size]
    y_train = dataset_y[:train_size]

    if test_size is None:
       return (x_train, y_train)
    # Create the test dataset
    x_test = dataset_x[train_size:train_size + test_size]
    y_test = dataset_y[train_size:train_size + test_size]

    return ((x_train, y_train), (x_test, y_test))


# In[ ]:


# # Assuming your original dataset is named dataset
# train_size = 900
# test_size = 100
# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{1000}_shuffled.pkl"
# dataset_file_name=simulation_directory_path+dataset_file_name
# print("dataset_file_name:",dataset_file_name)
# save_dataset(dataset_file_name,(dataset_x, dataset_y))

# dataset_file_name=""
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{1000}_shuffled_splitted.pkl"
# ((x_train, y_train), (x_test, y_test)) = split_dataset(dataset_x, dataset_y, train_size, test_size)

# # Display the shapes of the training and test datasets
# print("Training set shapes - x:", x_train.shape, " y:", y_train.shape)
# print("Test set shapes - x:", x_test.shape, " y:", y_test.shape)
# dataset_file_name=simulation_directory_path+dataset_file_name
# print("dataset_file_name:",dataset_file_name)
# save_dataset(dataset_file_name,((x_train, y_train), (x_test, y_test)))



# In[ ]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
def create_and_save_split_dataset(dataset_x,dataset_y,train_size,test_size,simulation_directory_path):
    dataset_size=train_size+test_size
    
    (x, y)=split_dataset(dataset_x, dataset_y, dataset_size)

    dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{dataset_size}_shuffled.pkl"
    dataset_file_name=simulation_directory_path+dataset_file_name
    print("dataset_file_name:",dataset_file_name)
    # Display the shapes of the training and test datasets
    print("Dataset shapes - x:", x.shape, " y:", y.shape)
    save_dataset(dataset_file_name,(x, y))

    ((x_train, y_train), (x_test, y_test)) = split_dataset(dataset_x, dataset_y, train_size, test_size)

    dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{dataset_size}_shuffled_splitted.pkl"
    dataset_file_name=simulation_directory_path+dataset_file_name
    print("dataset_file_name:",dataset_file_name)
    # Display the shapes of the training and test datasets
    print("Training set shapes - x:", x_train.shape, " y:", y_train.shape)
    print("Test set shapes - x:", x_test.shape, " y:", y_test.shape) 
    save_dataset(dataset_file_name,((x_train, y_train), (x_test, y_test)))


# In[ ]:


# Assuming your original dataset is named dataset
train_size = 900
test_size = 100
create_and_save_split_dataset(dataset_x,dataset_y,train_size,test_size,simulation_directory_path)

train_size = 9000
test_size = 1000
create_and_save_split_dataset(dataset_x,dataset_y,train_size,test_size,simulation_directory_path)

train_size = 90000
test_size = 10000
create_and_save_split_dataset(dataset_x,dataset_y,train_size,test_size,simulation_directory_path)

train_size = 900000
test_size = 100000
create_and_save_split_dataset(dataset_x,dataset_y,train_size,test_size,simulation_directory_path)


# In[ ]:


# train_size = 9000
# test_size = 1000

# ((x_train, y_train), (x_test, y_test)) = split_dataset(dataset_x, dataset_y, train_size, test_size)

# # Display the shapes of the training and test datasets
# print("Training set shapes - x:", x_train.shape, " y:", y_train.shape)
# print("Test set shapes - x:", x_test.shape, " y:", y_test.shape)

# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{10000}_shuffled.pkl"
# dataset_file_name=simulation_directory_path+dataset_file_name
# print("dataset_file_name:",dataset_file_name)
# save_dataset(dataset_file_name,((x_train, y_train), (x_test, y_test)))


# In[ ]:


# train_size = 90000
# test_size = 10000

# ((x_train, y_train), (x_test, y_test)) = split_dataset(dataset_x, dataset_y, train_size, test_size)

# # Display the shapes of the training and test datasets
# print("Training set shapes - x:", x_train.shape, " y:", y_train.shape)
# print("Test set shapes - x:", x_test.shape, " y:", y_test.shape)

# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{100000}_shuffled.pkl"
# dataset_file_name=simulation_directory_path+dataset_file_name
# print("dataset_file_name:",dataset_file_name)
# save_dataset(dataset_file_name,((x_train, y_train), (x_test, y_test)))


# In[ ]:


# train_size = 900000
# test_size = 100000

# ((x_train, y_train), (x_test, y_test)) = split_dataset(dataset_x, dataset_y, train_size, test_size)

# # Display the shapes of the training and test datasets
# print("Training set shapes - x:", x_train.shape, " y:", y_train.shape)
# print("Test set shapes - x:", x_test.shape, " y:", y_test.shape)

# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{1000000}_shuffled.pkl"
# dataset_file_name=simulation_directory_path+dataset_file_name
# print("dataset_file_name:",dataset_file_name)
# save_dataset(dataset_file_name,((x_train, y_train), (x_test, y_test)))

