#!/usr/bin/env python
# coding: utf-8

# # Point cloud classification with PointNet
# 
# **Author:** [David Griffiths](https://dgriffiths3.github.io)<br>
# **Date created:** 2020/05/25<br>
# **Last modified:** 2020/05/26<br>
# **Description:** Implementation of PointNet for ModelNet10 classification.

# # Point cloud classification
# 

# ## Introduction
# 
# Classification, detection and segmentation of unordered 3D point sets i.e. point clouds
# is a core problem in computer vision. This example implements the seminal point cloud
# deep learning paper [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593). For a
# detailed intoduction on PointNet see [this blog
# post](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a).
# 

# In[ ]:


import sys
sys.path.insert(1,'/wsu/home/gy/gy40/gy4065/hm.jetscapeml.source')
import jet_ml_dataset_builder.jet_ml_dataset_builder_utilities as util
util.install("trimesh")


# ## Setup
# 
# If using colab first install trimesh with `!pip install trimesh`.
# 

# In[ ]:



import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

tf.random.set_seed(1234)


# ## Load dataset
# 
# We use the ModelNet10 model dataset, the smaller 10 class version of the ModelNet40
# dataset. First download the data:
# 

# In[ ]:


# DATA_DIR = tf.keras.utils.get_file(
#     "modelnet.zip",
#     "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
#     extract=True,
# )
# DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")


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


# In[ ]:


print("Building required params for the loading the dataset file")

class_labels_str = '_'.join(y_class_label_items)
alpha_s_items_str='_'.join(map(str, alpha_s_items))
q0_items_str='_'.join(map(str, q0_items))
total_size=9*1200000
# for shuffled_y_processed
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_split_train_datasets/train_split_0.pkl"
# for shuffled
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{1000}_shuffled.pkl"
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{10000}_shuffled.pkl"
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{100000}_shuffled.pkl"
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{1000000}_shuffled.pkl"
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled.pkl"

dataset_file_name=simulation_directory_path+dataset_file_name
print("dataset_file_name:",dataset_file_name)


# In[ ]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import load_dataset
dataset=load_dataset(dataset_file_name,has_test=False)
# ((x_train, y_train),(x_test,y_test))=dataset
# dataset_x=[x_train,x_test]
# dataset_y=[x_test,y_test]
(dataset_x, dataset_y) = dataset
print("dataset.x:",type(dataset_x), dataset_x.size, dataset_x.shape)
print("dataset.y:",type(dataset_y), dataset_y.size,dataset_y.shape)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming dataset.x contains the images and dataset.y contains the labels
# Extract the first image and label
first_image = dataset_x[0]
label = dataset_y[0]

# Plot the first image with a white background
fig, ax = plt.subplots(figsize=(4, 4))

ax.imshow(first_image, cmap='gray', vmin=0, vmax=1)  # Assuming images are normalized between 0 and 1
# ax.set_title(f'Label: {label}')
ax.axis('off')
ax.set_facecolor('white')  # Set white background

# Save the plot as an image file (e.g., PNG)
plt.savefig('first_image_plot_labelless.png')

# Show the plot
plt.show()


# We can use the `trimesh` package to read and visualize the `.off` mesh files.
# 

# In[ ]:


# mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"))
# mesh.show()


# To convert a mesh file to a point cloud we first need to sample points on the mesh
# surface. `.sample()` performs a unifrom random sampling. Here we sample at 2048 locations
# and visualize in `matplotlib`.
# 

# In[ ]:


# points = mesh.sample(2048)

# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])
# ax.set_axis_off()
# plt.show()


# To generate a `tf.data.Dataset()` we need to first parse through the ModelNet data
# folders. Each mesh is loaded and sampled into a point cloud before being added to a
# standard python list and converted to a `numpy` array. We also store the current
# enumerate index value as the object label and use a dictionary to recall this later.
# 

# In[ ]:



# def parse_dataset(num_points=2048):

#     train_points = []
#     train_labels = []
#     test_points = []
#     test_labels = []
#     class_map = {}
#     folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

#     for i, folder in enumerate(folders):
#         print("processing class: {}".format(os.path.basename(folder)))
#         # store folder name with ID so we can retrieve later
#         class_map[i] = folder.split("/")[-1]
#         # gather all files
#         train_files = glob.glob(os.path.join(folder, "train/*"))
#         test_files = glob.glob(os.path.join(folder, "test/*"))

#         for f in train_files:
#             train_points.append(trimesh.load(f).sample(num_points))
#             train_labels.append(i)

#         for f in test_files:
#             test_points.append(trimesh.load(f).sample(num_points))
#             test_labels.append(i)

#     return (
#         np.array(train_points),
#         np.array(test_points),
#         np.array(train_labels),
#         np.array(test_labels),
#         class_map,
#     )



# Set the number of points to sample and batch size and parse the dataset. This can take
# ~5minutes to complete.
# 

# In[ ]:


# NUM_POINTS = 2048
# NUM_CLASSES = 10
# BATCH_SIZE = 32

# train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
#     NUM_POINTS
# )


# Our data can now be read into a `tf.data.Dataset()` object. We set the shuffle buffer
# size to the entire size of the dataset as prior to this the data is ordered by class.
# Data augmentation is important when working with point cloud data. We create a
# augmentation function to jitter and shuffle the train dataset.
# 

# In[ ]:



# def augment(points, label):
#     # jitter points
#     points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
#     # shuffle points
#     points = tf.random.shuffle(points)
#     return points, label


# train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

# train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
# test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)


# ### Build a model
# 
# Each convolution and fully-connected layer (with exception for end layers) consits of
# Convolution / Dense -> Batch Normalization -> ReLU Activation.
# 

# In[ ]:



# def conv_bn(x, filters):
#     x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
#     x = layers.BatchNormalization(momentum=0.0)(x)
#     return layers.Activation("relu")(x)


# def dense_bn(x, filters):
#     x = layers.Dense(filters)(x)
#     x = layers.BatchNormalization(momentum=0.0)(x)
#     return layers.Activation("relu")(x)



# PointNet consists of two core components. The primary MLP network, and the transformer
# net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
# network. The T-net is used twice. The first time to transform the input features (n, 3)
# into a canonical representation. The second is an affine transformation for alignment in
# feature space (n, 3). As per the original paper we constrain the transformation to be
# close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
# 

# In[ ]:



# class OrthogonalRegularizer(keras.regularizers.Regularizer):
#     def __init__(self, num_features, l2reg=0.001):
#         self.num_features = num_features
#         self.l2reg = l2reg
#         self.eye = tf.eye(num_features)

#     def __call__(self, x):
#         x = tf.reshape(x, (-1, self.num_features, self.num_features))
#         xxt = tf.tensordot(x, x, axes=(2, 2))
#         xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
#         return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))



#  We can then define a general function to build T-net layers.
# 

# In[ ]:



# def tnet(inputs, num_features):

#     # Initalise bias as the indentity matrix
#     bias = keras.initializers.Constant(np.eye(num_features).flatten())
#     reg = OrthogonalRegularizer(num_features)

#     x = conv_bn(inputs, 32)
#     x = conv_bn(x, 64)
#     x = conv_bn(x, 512)
#     x = layers.GlobalMaxPooling1D()(x)
#     x = dense_bn(x, 256)
#     x = dense_bn(x, 128)
#     x = layers.Dense(
#         num_features * num_features,
#         kernel_initializer="zeros",
#         bias_initializer=bias,
#         activity_regularizer=reg,
#     )(x)
#     feat_T = layers.Reshape((num_features, num_features))(x)
#     # Apply affine transformation to input features
#     return layers.Dot(axes=(2, 1))([inputs, feat_T])



# The main network can be then implemented in the same manner where the t-net mini models
# can be dropped in a layers in the graph. Here we replicate the network architecture
# published in the original paper but with half the number of weights at each layer as we
# are using the smaller 10 class ModelNet dataset.
# 

# In[ ]:


# inputs = keras.Input(shape=(NUM_POINTS, 3))

# x = tnet(inputs, 3)
# x = conv_bn(x, 32)
# x = conv_bn(x, 32)
# x = tnet(x, 32)
# x = conv_bn(x, 32)
# x = conv_bn(x, 64)
# x = conv_bn(x, 512)
# x = layers.GlobalMaxPooling1D()(x)
# x = dense_bn(x, 256)
# x = layers.Dropout(0.3)(x)
# x = dense_bn(x, 128)
# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

# model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
# model.summary()


# ### Train model
# 
# Once the model is defined it can be trained like any other standard classification model
# using `.compile()` and `.fit()`.
# 

# In[ ]:


# model.compile(
#     loss="sparse_categorical_crossentropy",
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     metrics=["sparse_categorical_accuracy"],
# )

# model.fit(train_dataset, epochs=20, validation_data=test_dataset)


# ## Visualize predictions
# 
# We can use matplotlib to visualize our trained model performance.
# 

# In[ ]:


# data = test_dataset.take(1)

# points, labels = list(data)[0]
# points = points[:8, ...]
# labels = labels[:8, ...]

# # run test data through model
# preds = model.predict(points)
# preds = tf.math.argmax(preds, -1)

# points = points.numpy()

# # plot points with predicted class and label
# fig = plt.figure(figsize=(15, 10))
# for i in range(8):
#     ax = fig.add_subplot(2, 4, i + 1, projection="3d")
#     ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
#     ax.set_title(
#         "pred: {:}, label: {:}".format(
#             CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
#         )
#     )
#     ax.set_axis_off()
# plt.show()

