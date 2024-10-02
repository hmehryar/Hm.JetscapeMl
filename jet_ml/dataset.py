print ("Dataset Preprocessor")
from jet_ml.config import Config
import pandas as pd
def get_label_items():
    # print ('Aggregatring all parameters values')
    eloss_items=['MMAT','MLBT']
    alpha_s_items=[0.2 ,0.3 ,0.4]
    q0_items=[1.5 ,2.0 ,2.5]
    data_dict = {
        "eloss_items": eloss_items,
        "alpha_s_items": alpha_s_items,
        "q0_items": q0_items
    }
    # print("label_items:\n",data_dict)
    return data_dict

def get_labels_str(label_items_dict=None):
  if label_items_dict==None:
      label_items_dict = get_label_items()
      return get_labels_str(label_items_dict)
#   print("Building required params for the loading the dataset file")

  data_dict = {
      "eloss_items_str":'_'.join(label_items_dict['eloss_items']),
      "alpha_s_items_str":'_'.join(map(str, label_items_dict['alpha_s_items'])),
      "q0_items_str":'_'.join(map(str, label_items_dict['q0_items'])),
  }
#   print("labels_str:\n",data_dict)
  return data_dict


def load_dataset(size: int, label_str_dict: dict=None, working_column: int = None, has_test: bool = False):
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
    
    dataset_file_name = Config().DATA_DIR / dataset_file_name

    print("Loading the whole dataset")
    dataset = pd.read_pickle(dataset_file_name)
    if has_test:
        (dataset_x_train, dataset_y_train), (dataset_x_test, dataset_y_test) = dataset
        print("dataset.x_train:",type(dataset_x_train), dataset_x_train.size, dataset_x_train.shape)
        print("dataset.y_train:",type(dataset_y_train), dataset_y_train.size,dataset_y_train.shape)

        print("dataset.x_test:",type(dataset_x_test), dataset_x_test.size, dataset_x_test.shape)
        print("dataset.y_test:",type(dataset_y_test), dataset_y_test.size, dataset_y_test.shape)
        del dataset
        if working_column is not None:
            print(f'Extract the working column#{working_column} for classification')
            dataset_y_train = dataset_y_train[:, working_column]
            dataset_y_test = dataset_y_test[:, working_column]
        return ((dataset_x_train, dataset_y_train), (dataset_x_test, dataset_y_test))
    else:
        (dataset_x, dataset_y) = dataset
        if working_column is not None:
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


def is_normalized(x):
    import numpy as np
    return np.max(x) <= 1

def convert_to_rgb(x):
    import numpy as np
    import tensorflow as tf
    # Convert grayscale to RGB (if needed)
    x_rgb = np.concatenate([x] * 3, axis=-1)  # Convert to RGB
    return x_rgb

def resize_images(x,width=32,height=32,device="/CPU:0"):
    import numpy as np
    import tensorflow as tf
    # Resize images to the target size
    TARGET_WIDTH = width
    TARGET_HEIGHT = height
    # with tf.device('/CPU:0'):
    x_resized = np.array([tf.image.resize(image, [TARGET_HEIGHT, TARGET_WIDTH]) for image in x])
    return x_resized
def resize_y(y):
    import numpy as np
    # Optionally, you might want to add an extra dimension if needed
    y_resized = np.expand_dims(y, axis=2)  # Shape will be (1, 3, 1000)
    return y_resized

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_train_data_generator(x_train, y_train, batch_size=32):
    """
    Create an ImageDataGenerator and return the training data generator.

    Parameters:
    - x_train: Array-like, training data features.
    - y_train: Array-like, training data labels.
    - batch_size: Integer, the size of the batches of data (default: 32).

    Returns:
    - train_generator: The configured ImageDataGenerator.
    """
    # Define ImageDataGenerator
    training_datagen = ImageDataGenerator(
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create the data generator
    train_generator = training_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=42  # For reproducibility
    )

    return train_generator


def create_validation_data_generator(x_validate, y_validate, batch_size=32):
    """
    Create an ImageDataGenerator for validation data and return the validation data generator.

    Parameters:
    - x_validate: Array-like, validation data features.
    - y_validate: Array-like, validation data labels.
    - batch_size: Integer, the size of the batches of data (default: 32).

    Returns:
    - val_generator: The configured ImageDataGenerator for validation.
    """
    # Define ImageDataGenerator for validation
    validation_datagen = ImageDataGenerator()

    # Create the validation data generator
    val_generator = validation_datagen.flow(
        x_validate,
        y_validate,
        batch_size=batch_size
    )

    return val_generator
