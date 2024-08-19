print ("Dataset Preprocessor")
from jet_ml import config
import pandas as pd
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


def load_dataset(size: int, label_str_dict: dict=None, working_column: int = 0):
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
    label_str_dict=get_labels_str()
    dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{label_str_dict['alpha_s_items_str']}_q0_{label_str_dict['q0_items_str']}_{label_str_dict['eloss_items_str']}_size_{size}_shuffled.pkl"

    dataset_file_name = config.DATA_DIR / dataset_file_name

    print("Loading the whole dataset")
    dataset = pd.read_pickle(dataset_file_name)
    (dataset_x, dataset_y) = dataset
    
    print(f'Extract the working column#{working_column} for classification')
    dataset_y = dataset_y[:, working_column]
    print("dataset.x:",type(dataset_x), dataset_x.size, dataset_x.shape)
    print("dataset.y:",type(dataset_y), dataset_y.size,dataset_y.shape)

    return dataset_x, dataset_y

from keras import backend as K
def reshape_x(x):
    img_rows,img_cols=32,32

    if K.image_data_format()=='channels_first':
        x=x.reshape(x.shape[0],1,img_rows,img_cols)
        # input_shape=(1,img_rows,img_cols)
    else:
        x=x.reshape(x.shape[0],img_rows,img_cols,1)
        # input_shape=(img_rows,img_cols,1)
    x=x.astype('float32')
    return x


def normalize_x(x):
    max=x.max()
    x/=max
    return x

import pandas as pd
def categorize_y(y_raw):
    dummies = pd.get_dummies(y_raw,dtype=int) # Classification
    # classes = dummies.columns
    
    # y = dummies.values
    # return (y,classes)
    return (y_raw, dummies)
