#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(1,'/wsu/home/gy/gy40/gy4065/hm.jetscapeml.source')

import numpy as np


# In[ ]:


print('Loading/Installing Package => Begin\n\n')
# from jet_ml_dataset_builder_utilities import install
# #reading/writing into files
# # !pip3 install pickle5
# install("pickle5")
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
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled.pkl"
dataset_file_name=simulation_directory_path+dataset_file_name
print("dataset_file_name:",dataset_file_name)


# In[ ]:


# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import load_dataset
# dataset=load_dataset (dataset_file_name)
# ((x_train,y_train),(x_test,y_test))=dataset
# print("dataset y_train values:\n", y_train[1:100])
# print("dataset y_test values:\n", y_test[1:10])


# # Processing y Labels

# In[ ]:



# from sklearn.preprocessing import OneHotEncoder, StandardScaler


# # Assuming you have the dataset stored in variables: x_train, x_test, y_train, y_test

# # Preprocess y_train and y_test
# # One-hot encode the categorical variable
# y_train_categorical = np.array(y_train[:, 0]).reshape(-1, 1)
# y_test_categorical = np.array(y_test[:, 0]).reshape(-1, 1)

# encoder = OneHotEncoder(sparse=False)
# y_train_categorical_encoded = encoder.fit_transform(y_train_categorical)
# y_test_categorical_encoded = encoder.transform(y_test_categorical)


# # Standardize the numerical variables
# scaler = StandardScaler()
# y_train_numerical = np.array(y_train[:, 1:])
# y_test_numerical = np.array(y_test[:, 1:])

# y_train_numerical_scaled = scaler.fit_transform(y_train_numerical)
# y_test_numerical_scaled = scaler.transform(y_test_numerical)

# # Combine the encoded categorical and scaled numerical columns
# y_train_processed = np.hstack((y_train_categorical_encoded, y_train_numerical_scaled))
# y_test_processed = np.hstack((y_test_categorical_encoded, y_test_numerical_scaled))


# In[ ]:


# y processed dataset file name
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled_y_processed.pkl"
dataset_file_name=simulation_directory_path+dataset_file_name


# In[ ]:


# print ("Saving the processed dataset")
# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset

# dataset_processed=((x_train,y_train_processed),(x_test,y_test_processed))
# save_dataset(dataset_file_name,dataset_processed)


# In[ ]:


from sklearn.model_selection import train_test_split

print ("Loading the processed dataset")
from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import load_dataset
# dataset=load_dataset(dataset_file_name)
# ((x_train,y_train_processed),(x_test,y_test_processed))=dataset
((x_train,y_train_processed),(x_test,y_test_processed)) = load_dataset(dataset_file_name)
print ("Loaded the processed dataset")
print("Now that you're done with dataset , release memory")
# del dataset


# In[ ]:


import numpy as np
import os

from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
# Assuming you have the training dataset as x_train_resized and y_train_processed
# Determine the number of pieces you want to split the dataset into
num_pieces = 64
batch_size = len(x_train) // num_pieces

# Create a directory to store the split datasets
output_directory = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_split_train_datasets/"
output_directory=simulation_directory_path+output_directory

os.makedirs(output_directory, exist_ok=True)

# Split the dataset and save each piece
for i in range(num_pieces):
    start = i * batch_size
    end = (i + 1) * batch_size if i < num_pieces - 1 else len(x_train)
    x_split = x_train[start:end]
    y_split = y_train_processed[start:end]

    # Save the split dataset
    print (f'Saving {output_directory}train_split_{i}.pkl')
    save_dataset(f'{output_directory}train_split_{i}.pkl',(x_split,y_split))

print("Training dataset split into 40 pieces and saved.")
exit()


# In[ ]:



# Split the data into training and validation sets
x_train_processed, x_val_processed, y_train_processed, y_val_processed = train_test_split(
    x_train, y_train_processed, test_size=0.1, random_state=42
)
# Print the shapes of processed data for verification
print("Shape of x_train_processed:", x_train_processed.shape)
print("Shape of y_train_processed:", y_train_processed.shape)
print("Shape of x_val_processed:", x_val_processed.shape)
print("Shape of y_val_processed:", y_val_processed.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test_processed:", y_test_processed.shape)

print("Now that you're done with x_train , release memory")
del x_train


# In[ ]:


# y processed x resized dataset file name
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled_y_processed_x_resized.pkl"
dataset_file_name=simulation_directory_path+dataset_file_name


# In[ ]:



# from skimage.transform import resize

# # Assuming you have the dataset stored in variables: x_train_processed, x_val_processed, x_test

# # Resize training set
# print("Resize training set")
# x_train_resized = np.array([resize(img, (224, 224, 3)) for img in x_train_processed])
# # x_train_processed=x_train_resized


# In[ ]:


# import tensorflow as tf
# from tensorflow.image import resize

# # Convert the NumPy arrays to TensorFlow Tensors
# x_train_processed = tf.convert_to_tensor(x_train_processed, dtype=tf.float32)
# # Resize training set
# print("Resize training set")
# x_train_resized = resize(x_train_processed, (224, 224), method='bilinear')
# x_train_resized = tf.image.grayscale_to_rgb(x_train_resized)  # Ensure it has 3 channels
# x_train_resized = x_train_resized.numpy()  # Convert back to NumPy array

# print("deleting x_trained_processoed")
# del x_train_processed


# In[ ]:


from skimage.transform import resize
#part1 0-4373995
#part2 4373996 -len(image_set)
# Define a function to generate resized images in batches
def batch_resized_images(image_set, batch_size, target_size):
    num_images = len(image_set)
    start = 0
    
    while start < num_images:
        end = min(start + batch_size, num_images)
        batch = image_set[start:end]
        resized_batch = [resize(img, target_size) for img in batch]
        yield np.array(resized_batch)
        start = end

# Resize training set
print("Resize training set")
# Parameters
batch_size = 1024  # Adjust this batch size according to your memory capacity
target_size = (224, 224, 3)
# Resize training set in batches
x_train_resized = []
for batch in batch_resized_images(x_train_processed, batch_size, target_size):
    x_train_resized.append(batch)
# Convert to a single NumPy array
x_train_resized = np.vstack(x_train_resized)   


# def resize_batch_images(num_images,start):
#     # Resize training set in batches
#     x_train_resized = []
#     for batch in batch_resized_images(x_train_processed, batch_size, target_size,num_images,start):
#         x_train_resized.append(batch)
#     # Convert to a single NumPy array
#     x_train_resized = np.vstack(x_train_resized)   
#     return x_train_resized



# In[ ]:



from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
def save_batched_x_resized_train(part,x_train_resized):
    # y processed x resized dataset file name
    dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled_x_resized_train_part_{part}.pkl"
    dataset_file_name=simulation_directory_path+dataset_file_name

    print("Saving dataset_x_resized_train")
    save_dataset(dataset_file_name,x_train_resized)
    print("Saved dataset_x_resized_train")

print("deleting x_trained_processoed")
del x_train_processed

del x_train_resized, y_train_processed

for part in 4:
    resize_batch_images()


# In[ ]:



# print("Saving dataset_processed_resized_train")
# dataset_processed_resized_train=(x_train_resized,y_train_processed)
# save_dataset(dataset_file_name,dataset_processed_resized_train)
# print("Saved dataset_processed_resized_train")
# del dataset_processed_resized_train, x_train_resized, y_train_processed


# In[ ]:


exit()


# In[ ]:


# # Resize validation set
# print("Resize validation set")
# x_val_resized = np.array([resize(img, (224, 224, 3)) for img in x_val_processed])
# # x_val_processed=x_val_resized


# In[ ]:


# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
# # y processed x resized dataset file name
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled_y_processed_x_resized_val.pkl"
# dataset_file_name=simulation_directory_path+dataset_file_name

# print("Saving dataset_processed_resized_val")
# dataset_processed_resized_val=(x_val_resized,y_val_processed)
# save_dataset(dataset_file_name,dataset_processed_resized_val)
# print("Saved dataset_processed_resized_val")
# del dataset_processed_resized_val,x_val_resized,y_val_processed,x_val_processed


# In[ ]:


from skimage.transform import resize
# Resize test set
print("Resize test set")
x_test_resized = np.array([resize(img, (224, 224, 3)) for img in x_test])
print("Deleting x_test")
del x_test
# x_test=x_test_resized


# In[ ]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled_y_processed_x_resized_test.pkl"
dataset_file_name=simulation_directory_path+dataset_file_name

print("Saving dataset processes and resized")
dataset_processed_resized_test=(x_test_resized,y_test_processed)
save_dataset(dataset_file_name,dataset_processed_resized_test)
print("Saved dataset processes and resized")
del dataset_processed_resized_test,x_test_resized,y_test_processed


# In[ ]:


exit()


# In[ ]:


# # a parallel implementation
# import tensorflow as tf
# from tensorflow.image import resize

# # Assuming you have the dataset stored in variables: x_train_processed, x_val_processed, x_test

# # Convert the NumPy arrays to TensorFlow Tensors
# x_train_processed = tf.convert_to_tensor(x_train_processed, dtype=tf.float32)
# x_val_processed = tf.convert_to_tensor(x_val_processed, dtype=tf.float32)
# x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

# # Resize training set
# print("Resize training set")
# x_train_resized = resize(x_train_processed, (224, 224), method='bilinear')
# x_train_resized = tf.image.grayscale_to_rgb(x_train_resized)  # Ensure it has 3 channels
# x_train_resized = x_train_resized.numpy()  # Convert back to NumPy array

# # Resize validation set
# print("Resize validation set")
# x_val_resized = resize(x_val_processed, (224, 224), method='bilinear')
# x_val_resized = tf.image.grayscale_to_rgb(x_val_resized)  # Ensure it has 3 channels
# x_val_resized = x_val_resized.numpy()  # Convert back to NumPy array

# # Resize test set
# print("Resize test set")
# x_test_resized = resize(x_test, (224, 224), method='bilinear')
# x_test_resized = tf.image.grayscale_to_rgb(x_test_resized)  # Ensure it has 3 channels
# x_test_resized = x_test_resized.numpy()  # Convert back to NumPy array


# In[ ]:


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
print("Saving dataset processes and resized")
dataset_processed_resized=((x_train_resized,y_train_processed),(x_val_resized,y_val_processed),(x_test_resized,y_test_processed))
save_dataset(dataset_file_name,dataset_processed_resized)
print("Saved dataset processes and resized")
del dataset_processed_resized
exit()


# 
# To build a deep neural model based on the VGG16 architecture for predicting the target parameters in the "y" side of your dataset, you can use transfer learning. Transfer learning involves using pre-trained models and fine-tuning them for your specific task.
# 
# In this case, we will use the pre-trained VGG16 model, remove its top layers (which are specific to the original classification task), and add new layers for our multi-task prediction. Since you have three target parameters in the "y" side, we will create three output layers, each predicting one of the target parameters.
# 

# In[ ]:



import numpy as np
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# In[ ]:


# # Replace these with your actual preprocessed dataset
# x_train_processed 
# x_val_processed 
# y_train_processed 
# y_val_processed 


# In[ ]:



def cnn_model(input_shape,lr):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(9, activation='softmax')(x)  # Assuming 9 total categories (2 for "MMAT", "MLBT" + 3 for "q0" + 4 for "alpha_s")
    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model


# In[ ]:





# In[ ]:


from os import path, makedirs
save_dir = simulation_directory_path+'simulation_result_vgg16_synthesis_10800k'
if not path.exists(save_dir):
    makedirs(save_dir)
print('Directory to save models: {}'.format(save_dir))


# In[ ]:


monitor='val_accuracy'


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def get_callbacks(monitor, save_dir):
    mode = None
    if 'loss' in monitor:
        mode = 'min'
    elif 'accuracy' in monitor:
        mode = 'max'
    assert mode != None, 'Check the monitor parameter!'

    # es = EarlyStopping(monitor=monitor, mode=mode, patience=10,
    #                   min_delta=0., verbose=1)
    es = EarlyStopping(monitor=monitor, mode=mode, patience=3, restore_best_weights=True)
    # rlp = ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.2, patience=5,
    #                         min_lr=0.001, verbose=1)
    mcp = ModelCheckpoint(path.join(save_dir, 'hm_jetscape_ml_model_best.h5'), monitor=monitor, 
                          save_best_only=True, mode=mode, verbose=1)
    
    # return [es, rlp, mcp]
    return [es, mcp]



# In[ ]:


# early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
import tensorflow as tf
callbacks = get_callbacks(monitor, save_dir)
n_epochs=30
batch_size=32
lr=0.001
input_shape=(32, 32, 3)

from time import time
def train_network(train_set, val_set, n_epochs, batch_size, monitor):
    tf.keras.backend.clear_session()
    X_train = train_set[0]
    Y_train = train_set[1]
    model = cnn_model(input_shape, lr)
    callbacks = get_callbacks(monitor, save_dir)

    model.summary()
    
    start = time()

    history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=val_set,
        callbacks=callbacks
        )
    train_time = (time()-start)/60.
    return history, train_time


# In[ ]:


train_set, val_set = (x_train_resized, y_train_processed), (x_val_resized, y_val_processed)
history, train_time = train_network(train_set, val_set, n_epochs, lr, batch_size, monitor)


# In[ ]:


import pandas as pd

file_name='hm_jetscape_ml_model_history.csv'
file_path=save_dir+file_name
pd.DataFrame.from_dict(history.history).to_csv(file_path,index=False)


file_name='hm_jetscape_ml_model_history.npy'
file_path=save_dir+file_name
np.save(file_path,history.history)


# In[ ]:



# Replace these with your actual test set
x_test_resized
y_test_processed 

test_loss, test_accuracy = model.evaluate(x_test_resized, y_test_processed)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# In[ ]:



# import numpy as np
# import matplotlib.pyplot as plt


# # Plot the loss and accuracy curves
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()

# # Save the plot as an image file
# plt.savefig('loss_accuracy_plot.png')

# plt.show()

