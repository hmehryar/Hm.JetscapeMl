#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install --upgrade tensorflow
import tensorflow as tf


# In[2]:


# #uncomment this cell when you are on COLAB
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
# from google.colab import drive
# drive.mount('/content/drive')


# In[3]:


import sys
sys.path.insert(1,'/wsu/home/gy/gy40/gy4065/hm.jetscapeml.source')
sys.path.insert(1,'/content/drive/My Drive/Projects/110_JetscapeMl/hm.jetscapeml.source')
sys.path.insert(1,'/content/drive/MyDrive/Projects/110_JetscapeMl/hm.jetscapeml.source')
sys.path.insert(1,'/g/My Drive/Projects/110_JetscapeMl/hm.jetscapeml.source')
sys.path.insert(1,'G:\\My Drive\\Projects\\110_JetscapeMl\\hm.jetscapeml.source')


# In[4]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import set_directory_paths
dataset_directory_path, simulation_directory_path = set_directory_paths()
from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import get_labels_str
label_str_dict=get_labels_str()


# In[5]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import get_dataset
dataset_size=1000
dataset_x, dataset_y=get_dataset(dataset_size,label_str_dict, dataset_directory_path)
from jet_ml_models.pointnet import preprocess_dataset
(dataset_x, dataset_y)=preprocess_dataset(dataset_x, dataset_y,is_one_hot_encoded=False,working_column=1,scale_x=True)


# In[6]:


from jet_ml_models.pointnet import prepare_datasets
from jet_ml_models.pointnet import augment
# Prepare datasets for training
train_dataset, validation_dataset,test_dataset = prepare_datasets(dataset_x, dataset_y, random_state=42,test_size=0.1, validation_size=None, augment=augment, batch_size=32)


# In[7]:


print(len(train_dataset))
print(len(validation_dataset))
print(len(test_dataset))


# In[8]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import generate_simulation_path
# monitor = 'val_accuracy'  # 'val_accuracy' or 'val_loss'
monitor="val_sparse_categorical_accuracy"



classifying_parameter="alpha_s"
fold = 1
n_epochs = 100

current_simulation_path = generate_simulation_path(simulation_directory_path, classifying_parameter,label_str_dict, dataset_size, n_epochs, fold)
print("current_simulation_path:",current_simulation_path)

# Use ModelCheckpoint callback to save the best model
best_model_file_path = f'{current_simulation_path}_best_model.keras'
print("best_model_file_path:",best_model_file_path)


# In[9]:


from jet_ml_models.pointnet import build_pointnet_classifier_model
NUM_POINTS = 1024
#because alpha_s can get 3 values
NUM_CLASSES = 3
activation="softmax"
# activation="sigmoid"
pointnet=build_pointnet_classifier_model(NUM_POINTS=NUM_POINTS,NUM_CLASSES=NUM_CLASSES, activation=activation)


# In[10]:


# !pip install tensorflow scikeras


# In[11]:


from jet_ml_models.pointnet import compile_pointnet_classifier_model_with_hyperparam

# learning_rate=0.001
loss='sparse_categorical_crossentropy'
# loss='categorical_crossentropy',
metrics='sparse_categorical_accuracy'
# metrics=['accuracy'],
pointnet=compile_pointnet_classifier_model_with_hyperparam(pointnet, loss=loss,metrics=metrics)
# from jet_ml_models.pointnet import print_model_summary
# print_model_summary(pointnet)


# In[12]:


from scikeras.wrappers import KerasClassifier
model = KerasClassifier(model=pointnet, verbose=0)

from sklearn.model_selection import GridSearchCV
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [10, 50, 100]
batch_size = [32]
epochs = [2, 5]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(dataset_x, dataset_y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

