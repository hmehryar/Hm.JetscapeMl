#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys


sys.path.insert(1,'/wsu/home/gy/gy40/gy4065/hm.jetscapeml.source')


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


from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import load_dataset
dataset=load_dataset (dataset_file_name)
((x_train,y_train),(x_test,y_test))=dataset
print("dataset y_train values:\n", y_train[1:100])
print("dataset y_test values:\n", y_test[1:10])


# # Processing y Labels

# In[ ]:


import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Assuming you have the dataset stored in variables: x_train, x_test, y_train, y_test

# Preprocess y_train and y_test
# One-hot encode the categorical variable
y_train_categorical = np.array(y_train[:, 0]).reshape(-1, 1)
y_test_categorical = np.array(y_test[:, 0]).reshape(-1, 1)

encoder = OneHotEncoder(sparse=False)
y_train_categorical_encoded = encoder.fit_transform(y_train_categorical)
y_test_categorical_encoded = encoder.transform(y_test_categorical)


# Standardize the numerical variables
scaler = StandardScaler()
y_train_numerical = np.array(y_train[:, 1:])
y_test_numerical = np.array(y_test[:, 1:])

y_train_numerical_scaled = scaler.fit_transform(y_train_numerical)
y_test_numerical_scaled = scaler.transform(y_test_numerical)

# Combine the encoded categorical and scaled numerical columns
y_train_processed = np.hstack((y_train_categorical_encoded, y_train_numerical_scaled))
y_test_processed = np.hstack((y_test_categorical_encoded, y_test_numerical_scaled))

print ("Saving the processed dataset")
from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import save_dataset
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled_y_processed.pkl"
dataset_file_name=simulation_directory_path+dataset_file_name
dataset_processed=((x_train,y_train_processed),(x_test,y_test_processed))
save_dataset(dataset_file_name,dataset_processed)


# Split the data into training and validation sets
x_train_processed, x_val_processed, y_train_processed, y_val_processed = train_test_split(
    x_train, y_train_processed, test_size=0.1, random_state=42
)

# Print the shapes of processed data for verification
print("Shape of x_train_processed:", x_train_processed.shape)
print("Shape of x_val_processed:", x_val_processed.shape)
print("Shape of y_train_processed:", y_train_processed.shape)
print("Shape of y_val_processed:", y_val_processed.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test_processed:", y_test_processed.shape)


# In[ ]:



# from skimage.transform import resize

# # Assuming you have the dataset stored in variables: x_train_processed, x_val_processed, x_test

# # Resize training set
# x_train_resized = np.array([resize(img, (224, 224, 3)) for img in x_train_processed])
# # x_train_processed=x_train_resized
# # Resize validation set
# x_val_resized = np.array([resize(img, (224, 224, 3)) for img in x_val_processed])
# # x_val_processed=x_val_resized
# # Resize test set
# x_test_resized = np.array([resize(img, (224, 224, 3)) for img in x_test])
# # x_test=x_test_resized


# 
# To build a deep neural model based on the VGG16 architecture for predicting the target parameters in the "y" side of your dataset, you can use transfer learning. Transfer learning involves using pre-trained models and fine-tuning them for your specific task.
# 
# In this case, we will use the pre-trained VGG16 model, remove its top layers (which are specific to the original classification task), and add new layers for our multi-task prediction. Since you have three target parameters in the "y" side, we will create three output layers, each predicting one of the target parameters.
# 

# In[ ]:



# import numpy as np
# from keras.applications import VGG16
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping


# In[ ]:


# # Replace these with your actual preprocessed dataset
# x_train_processed 
# x_val_processed 
# y_train_processed 
# y_val_processed 


# In[ ]:


# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))


# In[ ]:



# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# predictions = Dense(9, activation='softmax')(x)  # Assuming 9 total categories (2 for "MMAT", "MLBT" + 3 for "q0" + 4 for "alpha_s")
# model = Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


# model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

# history = model.fit(
#     x_train_resized, y_train_processed,
#     batch_size=32,
#     epochs=20,
#     validation_data=(x_val_resized, y_val_processed),
#     callbacks=[early_stopping]
# )



# In[ ]:



# # Replace these with your actual test set
# x_test_resized
# y_test_processed 

# test_loss, test_accuracy = model.evaluate(x_test_resized, y_test_processed)
# print("Test Loss:", test_loss)
# print("Test Accuracy:", test_accuracy)


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

