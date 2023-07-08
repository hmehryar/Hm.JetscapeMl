# Commonly used modules
import sys
import subprocess
import numpy as np
import time
from time import time

def install(package):
  print("Installing "+package) 
  subprocess.check_call([sys.executable,"-m" ,"pip", "install", package])
  print("Installed "+package+"\n") 

#reading/writing into files
# !pip3 install pickle5
install("pickle5")
import pickle5 as pickle

def save_dataset(file_name,dataset):
    with open(file_name, 'wb') as dataset_file:
        pickle.dump(dataset,dataset_file, protocol=pickle.HIGHEST_PROTOCOL)

        
def load_dataset(file_name):
    with open(file_name, 'rb') as dataset_file:
        (x_train, y_train), (x_test, y_test) = pickle.load(dataset_file, encoding='latin1')
        dataset=((x_train, y_train), (x_test, y_test))
        return dataset

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
    print(dataset_y_test[1:100])
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

    # Assuming you have already loaded your dataset into the following variables:
    # dataset.x_train, dataset.x_test, dataset.y_train, dataset.y_test
    ((dataset_x_train,dataset_y_train),(dataset_x_test,dataset_y_test))=dataset
    # Create a new column with a value of 0.4
    alpha_s_train_column = np.full(dataset_y_train.shape, alpha_s)
    alpha_s_test_column = np.full(dataset_y_test.shape, alpha_s)

    q0_train_column = np.full(dataset_y_train.shape, q0)
    q0_test_column = np.full(dataset_y_test.shape, q0)

    # Concatenate the new column with the target variable
    y_train_with_columns = np.column_stack((dataset_y_train, alpha_s_train_column,q0_train_column))
    y_test_with_columns = np.column_stack((dataset_y_test, alpha_s_test_column,q0_test_column))

    # Print the updated shapes and values of the resulting two columns
    print("Updated y_train shape:", y_train_with_columns.shape)
    print("Updated y_test shape:", y_test_with_columns.shape)
    print("Updated y_train values:\n", y_train_with_columns[1:100])
    print("Updated y_test values:\n", y_test_with_columns[1:100])

    return ((dataset_x_train,y_train_with_columns),(dataset_x_test,y_test_with_columns))