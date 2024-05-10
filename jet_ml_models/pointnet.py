import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return {'num_features': self.num_features, 'l2reg': self.l2reg}

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def build_pointnet_classifier_model(NUM_POINTS,NUM_CLASSES):
    inputs = keras.Input(shape=(NUM_POINTS, 3))
    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    # outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    return model

def print_model_summary(model):
    """
    Print model summary including metrics, loss, and optimizer method.
    
    Parameters:
    - model (tf.keras.Model): The TensorFlow model to print the summary for.
    """
    # Print model summary
    model.summary()
    
    # Print optimizer
    print("Optimizer: ", model.optimizer)
    
    # Print loss
    print("Loss function: ", model.loss)
    
    # Print metrics
    print("Metrics: ", model.metrics_names)



def compile_pointnet_classifier_model_with_hyperparam(model,learning_rate):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        # loss="sparse_categorical_crossentropy",
        loss='categorical_crossentropy',
        # loss="binary_crossentropy",
        optimizer=optimizer,
        # metrics=["sparse_categorical_accuracy"],
        metrics=['accuracy'],
    )
    print_model_summary(model)
    return model

def get_coordinates(image_array):
  import numpy as np
  # Get the dimensions of the original array
  height, width = image_array.shape
  # Create an array of coordinates (x, y)
  coordinates = np.column_stack((np.repeat(np.arange(height), width),
                                np.tile(np.arange(width), height)))
  return coordinates
def get_point_clouds(image_array,coordinates):
  # Assuming image_array is your 32x32 numpy array
  # image_array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
  # Create an nx3 array with x, y, and intensity values
  result_array = np.column_stack((coordinates, image_array.flatten()))
  return result_array

import numpy as np

def get_dataset_points(dataset_x):
    """
    Get 3D points for each entry in the dataset.

    Parameters:
    - dataset: 3D array-like, the dataset containing non-zero values.

    Returns:
    - dataset_points: NumPy array, each entry corresponds to the 3D points of non-zero values for a particular entry in the dataset.

    """
    dataset_points = []
    coordinates=get_coordinates(dataset_x[0])
    for data in dataset_x:

        point_clouds=get_point_clouds(data,coordinates)
        # Append coordinates to the list
        dataset_points.append(point_clouds)

    # Convert the list of coordinates to a NumPy array
    dataset_points = np.array(dataset_points)
    return dataset_points

def split_dataset(dataset_x_points, dataset_y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - dataset_x_points: The 3D coordinates corresponding to each entry.
    - dataset_y: The target values (2D array).
    - test_size: The proportion of the dataset to include in the test split.
    - random_state: Seed for random number generation.

    Returns:
    - x_train_points, x_test_points: The split 3D coordinates for training and testing.
    - y_train, y_test: The split target values for training and testing.
    """
    # @h.mehryar: I think this was redundant, not sure why I impllemented it, there is
    # no need to flatten the dataset_x for splitting it
    # Flatten the input data to 2D
    #flattened_dataset_x = dataset_x.reshape(dataset_x.shape[0], -1)
    
    # Split the dataset
    # x_train, x_test, x_train_points, x_test_points, y_train, y_test = \
    #     train_test_split(flattened_dataset_x, dataset_x_points, dataset_y, test_size=test_size, random_state=random_state)
    
    x_train_points, x_test_points, y_train, y_test = \
        train_test_split(dataset_x_points, dataset_y, test_size=test_size, random_state=random_state)

    # @h.mehryar: I think this was redundant, not sure why I impllemented it
    # # Reshape the input data back to 3D
    # x_train = x_train.reshape(x_train.shape[0], dataset_x.shape[1], dataset_x.shape[2])
    # x_test = x_test.reshape(x_test.shape[0], dataset_x.shape[1], dataset_x.shape[2])

    return x_train_points, x_test_points, y_train, y_test

def preprocess_dataset(dataset_x, dataset_y):
  print("Pre-processing")
  # Example usage:
  dataset_x_points = get_dataset_points(dataset_x)
  print("dataset_x_points shape:", dataset_x_points.shape)
  x_train_points, x_test_points, y_train, y_test= \
    split_dataset(dataset_x_points, dataset_y, test_size=0.2, random_state=None)
  print("deleting the original dataset after splitting ...")
  del dataset_x,dataset_x_points,dataset_y
  
  print("train_points:",type(x_train_points), x_train_points.size, x_train_points.shape)
  print("train_y:",type(y_train), y_train.size,y_train.shape)

  
  print("x_test_points:",type(x_test_points), x_test_points.size, x_test_points.shape)
  print("y_test:",type(y_test), y_test.size,y_test.shape)
  

  print("Preprocess y_train and y_test")
  print("One-hot encode the categorical variable")
  print("y_test:\n",y_test[:10])
  y_train_categorical = np.array(y_train).reshape(-1, 1)
  y_test_categorical = np.array(y_test).reshape(-1, 1)

  encoder = OneHotEncoder(sparse_output=False)
  y_train_categorical_encoded = encoder.fit_transform(y_train_categorical)
  y_test_categorical_encoded = encoder.transform(y_test_categorical)
  print("y_test_categorical:\n",y_test_categorical[:10])
  print("y_test_categorical_encoded:\n",y_test_categorical_encoded[:10])

  return (x_train_points,  y_train_categorical_encoded,x_test_points,  y_test_categorical_encoded)




