#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/hmehryar/Hm.JetscapeMl/blob/309-implementingtraining-pointnet-for-alpha_s-with-various-epochs-and-folds-and-finding-the-best-learning-rate/jet_ml_pointnet_alpha_s/jet_ml_pointnet_alpha_s.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


# !pip install --upgrade tensorflow
import tensorflow as tf


# In[2]:


#uncomment this cell when you are on COLAB
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


# In[5]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import get_labels_str
label_str_dict=get_labels_str()


# In[6]:


with tf.device("CPU"):
    from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import get_dataset
    dataset_size=1000000
    dataset_x, dataset_y=get_dataset(dataset_size,label_str_dict, dataset_directory_path,working_column=1,scale_x=True)


# In[7]:


with tf.device("CPU"):
    from jet_ml_models.pointnet import preprocess_dataset
    (x_train,  y_train,x_test,  y_test)=preprocess_dataset(dataset_x, dataset_y,is_one_hot_encoded=False)
    print("deleting original dataset")
    del dataset_x,dataset_y


# In[8]:


with tf.device("CPU"):
    from jet_ml_models.pointnet import create_tf_dataset
    print("converting to tensor data")
    # Create TensorFlow Dataset for training data and test data
    dataset = create_tf_dataset(x_train, y_train)
    test_dataset = create_tf_dataset(x_test, y_test)
    len_x_train=len(x_train)
    len_x_test=len(x_test)
    print("deleting preprocessed dataset")
    del x_train,y_train,x_test,y_test


# In[9]:


from jet_ml_models.pointnet import prepare_datasets
from jet_ml_models.pointnet import augment

with tf.device("CPU"):
    # Prepare datasets for training
    print("going to prepare dataset")
    train_dataset, validation_dataset,test_dataset = prepare_datasets(dataset, test_dataset, len_x_train, len_x_test, augment,train_size=1)
    print("deleting initial tensor dataset")
    del dataset


# In[10]:


with tf.device("CPU"):
    print(len(test_dataset))
    print(len(validation_dataset))


# In[11]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import generate_simulation_path
monitor = 'val_accuracy'  # 'val_accuracy' or 'val_loss'

classifying_parameter="alpha_s"
n_epochs = 100
fold = 1

current_simulation_path = generate_simulation_path(simulation_directory_path, classifying_parameter,label_str_dict, dataset_size, n_epochs, fold)
print("current_simulation_path:",current_simulation_path)

# Use ModelCheckpoint callback to save the best model
best_model_file_path = f'{current_simulation_path}_best_model.keras'
print("best_model_file_path:",best_model_file_path)


# In[12]:


from jet_ml_models.pointnet import build_pointnet_classifier_model

NUM_POINTS = 1024
#because alpha_s can get 3 values
NUM_CLASSES = 3
activation="softmax"
# activation="sigmoid"

pointnet=build_pointnet_classifier_model(NUM_POINTS=NUM_POINTS,NUM_CLASSES=NUM_CLASSES, activation=activation)


# In[13]:


from jet_ml_models.pointnet import compile_pointnet_classifier_model_with_hyperparam
from jet_ml_models.pointnet import print_model_summary
learning_rate=0.001
loss='sparse_categorical_crossentropy'
# loss='categorical_crossentropy',

metrics='sparse_categorical_accuracy'
# metrics=['accuracy'],

pointnet=compile_pointnet_classifier_model_with_hyperparam(pointnet,learning_rate=learning_rate, loss=loss,metrics=metrics)
# print_model_summary(pointnet)


# In[14]:


# %%timeit -n1 -r1
from jet_ml_models.pointnet import train_model_with_callbacks
monitor='val_sparse_categorical_accuracy'
with tf.device('/GPU:0'):
  model, history, train_time=train_model_with_callbacks(pointnet, train_dataset=train_dataset, validation_dataset=validation_dataset,monitor=monitor, best_model_file_path=best_model_file_path, n_epochs=n_epochs)


# In[15]:


print(train_time)
print(model)
print(history.history)


# In[16]:


from jet_ml_models.pointnet import save_training_history
training_history_file_path_json,training_history_file_path_csv,training_history_file_path_csv =   save_training_history(history,current_simulation_path)


# In[17]:


from jet_ml_models.pointnet import plot_training_history
plot_training_history_path=plot_training_history(history,current_simulation_path)
print(plot_training_history_path)


# In[18]:


from jet_ml_models.pointnet import evaluate_model
accuracy, confusion_matrix = evaluate_model(model,test_dataset=test_dataset)


# In[19]:


from jet_ml_models.pointnet import save_kfold_results
results_kfold = []
results_kfold.append({
                  'Dataset Size': dataset_size,
                  'Classifier': model.name,
                  'Fold Number': fold,
                  'Accuracy': accuracy,
                  'Confusion Matrix': confusion_matrix,
                  'Train Time': train_time,
                  'Loss/Accuracy Plot Path': plot_training_history_path,
                  'Best Model Path': best_model_file_path
              })
save_kfold_results(results_kfold, current_simulation_path)

