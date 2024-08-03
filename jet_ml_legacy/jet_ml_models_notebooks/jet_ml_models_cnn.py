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
dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{1000}_shuffled.pkl"
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{10000}_shuffled.pkl"
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{100000}_shuffled.pkl"
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{1000000}_shuffled.pkl"
# dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{total_size}_shuffled.pkl"

dataset_file_name=simulation_directory_path+dataset_file_name
print("dataset_file_name:",dataset_file_name)


# In[ ]:


# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import load_dataset
# dataset=load_dataset(dataset_file_name,has_test=False)
# # ((x_train, y_train),(x_test,y_test))=dataset
# # dataset_x=[x_train,x_test]
# # dataset_y=[x_test,y_test]
# (dataset_x, dataset_y) = dataset
# print("dataset.x:",type(dataset_x), dataset_x.size, dataset_x.shape)
# print("dataset.y:",type(dataset_y), dataset_y.size,dataset_y.shape)
# # print("dataset y_train values:\n", dataset_x[1:10])
# print("dataset y_test values:\n", dataset_y[1:10])


# - The first column of `dataset_y` is extracted (`dataset_y_binary`) for binary classification.
# - The dataset is split into training and testing sets using `train_test_split`.

# In[ ]:


# # Assuming x and y are defined
# # x should be a 2D array (e.g., (1000, 32*32))
# # y should be a 2D array with three columns (e.g., (1000, 3))

# from sklearn.model_selection import train_test_split

# test_size = 0.1  # Adjust the test_size as needed
# # Extract the first column for binary classification
# dataset_y_binary = dataset_y[:, 0]

# # Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y_binary, test_size=test_size, random_state=42)

# # Display the shapes of the training and test datasets
# print("Training set shapes - x:", x_train.shape, " y:", y_train.shape)
# print("Test set shapes - x:", x_test.shape, " y:", y_test.shape)


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import load_dataset
# Function to load datasets of different sizes
def get_dataset(size):

    dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{size}_shuffled.pkl"
    
    dataset_file_name=simulation_directory_path+dataset_file_name
    print("dataset_file_name:",dataset_file_name)
    
    dataset=load_dataset(dataset_file_name,has_test=False)
    (dataset_x, dataset_y) = dataset
    # Extract the first column for binary classification
    dataset_y = dataset_y[:, 0]
    print("dataset.x:",type(dataset_x), dataset_x.size, dataset_x.shape)
    print("dataset.y:",type(dataset_y), dataset_y.size,dataset_y.shape)
    return dataset_x, dataset_y

# Function to train and evaluate classifiers
def train_and_evaluate_classifier(model, x_train, y_train, x_test, y_test):
    # Compile the model
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_split=0.2)

    # Evaluate the model on the test set
    y_pred_probs = model.predict(x_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Assuming threshold of 0.5 for binary classification

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

# # Sizes of datasets
dataset_sizes = [1000]
# dataset_sizes = [1000, 10000]
# dataset_sizes = [1000, 10000, 100000, 1000000]

def model_cnn():
    # Build the CNN model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification, so using sigmoid activation
    ])
    
    return model
# Classifiers
classifiers = {
    'CNN': model_cnn(),
}

# Results storage
results = []

# Loop through different dataset sizes
for size in dataset_sizes:
    # Generate dataset
    x, y = get_dataset(size)
    

    # Assuming dataset_y is a NumPy array or a Pandas Series with string labels
    label_encoder = LabelEncoder()
    dataset_y_encoded = label_encoder.fit_transform(y)

    # Print the mapping of original labels to encoded labels
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label Mapping:", label_mapping)

    # Now, dataset_y_encoded can be used for training your CNN

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Calculate the maximum value in the training set
    max_value = np.max(x_train)

    # Normalize the pixel values to be between 0 and 1
    x_train, x_test = x_train / max_value, x_test / max_value

    # Loop through classifiers
    for clf_name, clf in classifiers.items():
        # Train and evaluate classifier
        accuracy, cm = train_and_evaluate_classifier(clf, x_train, y_train, x_test, y_test)
        results.append({'Dataset Size': size, 'Classifier': clf_name, 'Accuracy': accuracy, 'Confusion Matrix': cm})
        # Create a DataFrame from results
        df_results = pd.DataFrame(results)

        # Save the DataFrame to a text file
        df_results.to_csv('binary_classification_results.txt', index=False, sep='\t')


        # Plotting with different markers for each classifier
        plt.figure(figsize=(12, 8))
        markers = ['o', 's', '^', 'D', 'v']  # You can customize the markers here

        for clf_name, group, marker in zip(classifiers.keys(), df_results.groupby('Classifier'), markers):
            plt.plot(group[1]['Dataset Size'], group[1]['Accuracy'], label=clf_name, marker=marker)

        # # Plotting
        # plt.figure(figsize=(12, 8))
        # for clf_name, group in df_results.groupby('Classifier'):
        #     plt.plot(group['Dataset Size'], group['Accuracy'], label=clf_name, marker='o')

        plt.title('Binary Classification Accuracy for Different Dataset Sizes')
        plt.xlabel('Dataset Size')
        plt.xscale('log')
        plt.ylabel('Accuracy')
        plt.legend()

        # Save the plot with high resolution (300 dpi)
        plt.savefig('binary_classification_accuracy_plot.png', dpi=300)
        plt.show()

        # Display results in a table
        print(df_results.pivot_table(index='Dataset Size', columns='Classifier', values='Accuracy'))

        # Define the module labels
        module_labels = ['MMATTER', 'MLBT']

        # Save confusion matrices
        for index, row in df_results.iterrows():
            clf_name = row['Classifier']
            dataset_size = row['Dataset Size']
            cm = row['Confusion Matrix']
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap='Oranges') #plt.cm.Blue

            # Annotate each cell with the value
            for i in range(len(module_labels)):
                for j in range(len(module_labels)):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

            plt.title(f'Confusion Matrix - {clf_name} - {dataset_size} samples')
            plt.colorbar()
            # Set tick labels
            plt.xticks(np.arange(len(module_labels)), module_labels)
            plt.yticks(np.arange(len(module_labels)), module_labels)

            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            # Remove tick marks
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            plt.savefig(f'confusion_matrix_{clf_name}_{dataset_size}.png', dpi=300, bbox_inches='tight')
            plt.show()



