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


# # Assuming x and y are defined
# # x should be a 2D array (e.g., (1000, 32*32))
# # y should be a 2D array with three columns (e.g., (1000, 3))

# # Flatten the 32x32 images to 1D arrays for LogisticRegression, DecisionTreeClassifier, LinearSVM, KNN, RandomForests
# x_train_flatten = x_train.reshape(x_train.shape[0], -1)
# x_test_flatten = x_test.reshape(x_test.shape[0], -1)


# - The logistic regression model is trained specifically for binary classification on the first column.
# - Predictions and evaluation are performed based on the binary labels.

# In[ ]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Initialize the Logistic Regression model
# model_binary = LogisticRegression(max_iter=1000, random_state=42)

# # Train the model for binary classification
# model_binary.fit(x_train_flatten, y_train)

# # Make predictions on the test set
# y_pred_binary = model_binary.predict(x_test_flatten)

# # Evaluate the accuracy for binary classification
# accuracy_binary = accuracy_score(y_test, y_pred_binary)
# print(f"Accuracy (Binary Classification with Logistic Regression): {accuracy_binary}")


# This code uses DecisionTreeClassifier instead of LogisticRegression. The structure is similar: extract the first column for binary classification, split the dataset, flatten the images, initialize the model, train the model, make predictions, and evaluate the accuracy. 

# In[ ]:


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Initialize the Decision Tree Classifier model
# model_decision_tree = DecisionTreeClassifier(random_state=42)

# # Train the model for binary classification
# model_decision_tree.fit(x_train_flatten, y_train)

# # Make predictions on the test set
# y_pred_binary = model_decision_tree.predict(x_test_flatten)

# # Evaluate the accuracy for binary classification
# accuracy_binary = accuracy_score(y_test, y_pred_binary)
# print(f"Accuracy (Binary Classification with Decision Tree): {accuracy_binary}")


# This code uses LinearSVC instead of LogisticRegression or DecisionTreeClassifier. The structure remains similar: extract the first column for binary classification, split the dataset, flatten the images, initialize the model, train the model, make predictions, and evaluate the accuracy. 

# In[ ]:


# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score

# # Initialize the Linear Support Vector Classification model
# model_linear_svc = LinearSVC(random_state=42)

# # Train the model for binary classification
# model_linear_svc.fit(x_train_flatten, y_train)

# # Make predictions on the test set
# y_pred_binary = model_linear_svc.predict(x_test_flatten)

# # Evaluate the accuracy for binary classification
# accuracy_binary = accuracy_score(y_test, y_pred_binary)
# print(f"Accuracy (Binary Classification with LinearSVC): {accuracy_binary}")


# Adjust the k_neighbors parameter based on your requirements. The structure is similar to the previous examples: extract the first column for binary classification, split the dataset, flatten the images, initialize the model, train the model, make predictions, and evaluate the accuracy.

# In[ ]:


# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# # Initialize the KNN classifier model
# k_neighbors = 5  # You can adjust this parameter
# model_knn = KNeighborsClassifier(n_neighbors=k_neighbors)

# # Train the model for binary classification
# model_knn.fit(x_train_flatten, y_train)

# # Make predictions on the test set
# y_pred_binary = model_knn.predict(x_test_flatten)

# # Evaluate the accuracy for binary classification
# accuracy_binary = accuracy_score(y_test, y_pred_binary)
# print(f"Accuracy (Binary Classification with KNN): {accuracy_binary}")


# This code uses RandomForestClassifier from scikit-learn. The structure is similar to the previous examples: extract the first column for binary classification, split the dataset, flatten the images, initialize the model, train the model, make predictions, and evaluate the accuracy. 

# In[ ]:


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Initialize the Random Forest Classifier model
# model_random_forest = RandomForestClassifier(random_state=42)

# # Train the model for binary classification
# model_random_forest.fit(x_train_flatten, y_train)

# # Make predictions on the test set
# y_pred_binary = model_random_forest.predict(x_test_flatten)

# # Evaluate the accuracy for binary classification
# accuracy_binary = accuracy_score(y_test, y_pred_binary)
# print(f"Accuracy (Binary Classification with RandomForest): {accuracy_binary}")


# In[ ]:


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.model_selection import train_test_split

# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import load_dataset
# # Function to load datasets of different sizes
# def get_dataset(size):
#     # x = np.random.random((size, 32, 32))
#     # y = np.random.randint(0, 2, size=(size, 3))  # Assuming three columns for y
#     dataset_file_name = f"jet_ml_benchmark_config_01_to_09_alpha_{alpha_s_items_str}_q0_{q0_items_str}_{class_labels_str}_size_{size}_shuffled.pkl"
    
#     dataset_file_name=simulation_directory_path+dataset_file_name
#     print("dataset_file_name:",dataset_file_name)
    
#     dataset=load_dataset(dataset_file_name,has_test=False)
#     (dataset_x, dataset_y) = dataset
#     # Extract the first column for binary classification
#     dataset_y = dataset_y[:, 0]
#     print("dataset.x:",type(dataset_x), dataset_x.size, dataset_x.shape)
#     print("dataset.y:",type(dataset_y), dataset_y.size,dataset_y.shape)
#     return dataset_x, dataset_y

# # Function to train and evaluate classifiers
# def train_and_evaluate_classifier(model, x_train, y_train, x_test, y_test):
#     # Assuming x and y are defined
#     # x should be a 2D array (e.g., (1000, 32*32))
#     # y should be a 2D array with three columns (e.g., (1000, 3))

#     # Flatten the 32x32 images to 1D arrays for LogisticRegression, DecisionTreeClassifier, LinearSVM, KNN, RandomForests
#     x_train_flatten = x_train.reshape(x_train.shape[0], -1)
#     x_test_flatten = x_test.reshape(x_test.shape[0], -1)
#     model.fit(x_train_flatten, y_train)
#     y_pred = model.predict(x_test_flatten)
#     accuracy = accuracy_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     return accuracy, cm


# # # Sizes of datasets
# # dataset_sizes = [1000, 10000]
# dataset_sizes = [1000, 10000, 100000, 1000000]


# # Classifiers
# classifiers = {
#     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#     'Decision Tree': DecisionTreeClassifier(random_state=42),
#     'LinearSVC': LinearSVC(random_state=42),
#     'KNN': KNeighborsClassifier(),
#     'Random Forest': RandomForestClassifier(random_state=42)
# }

# # Results storage
# results = []

# # Loop through different dataset sizes
# for size in dataset_sizes:
#     # Generate dataset
#     x, y = get_dataset(size)
    
#     # Split dataset
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
#     # Loop through classifiers
#     for clf_name, clf in classifiers.items():
#         # Train and evaluate classifier
#         accuracy, cm = train_and_evaluate_classifier(clf, x_train, y_train, x_test, y_test)
#         results.append({'Dataset Size': size, 'Classifier': clf_name, 'Accuracy': accuracy, 'Confusion Matrix': cm})

# # Create a DataFrame from results
# df_results = pd.DataFrame(results)

# # Save the DataFrame to a text file
# df_results.to_csv('binary_classification_results.txt', index=False, sep='\t')


# # Plotting with different markers for each classifier
# plt.figure(figsize=(12, 8))
# markers = ['o', 's', '^', 'D', 'v']  # You can customize the markers here

# for clf_name, group, marker in zip(classifiers.keys(), df_results.groupby('Classifier'), markers):
#     plt.plot(group[1]['Dataset Size'], group[1]['Accuracy'], label=clf_name, marker=marker)

# # # Plotting
# # plt.figure(figsize=(12, 8))
# # for clf_name, group in df_results.groupby('Classifier'):
# #     plt.plot(group['Dataset Size'], group['Accuracy'], label=clf_name, marker='o')

# plt.title('Binary Classification Accuracy for Different Dataset Sizes')
# plt.xlabel('Dataset Size')
# plt.xscale('log')
# plt.ylabel('Accuracy')
# plt.legend()

# # Save the plot with high resolution (300 dpi)
# plt.savefig('binary_classification_accuracy_plot.png', dpi=300)
# plt.show()

# # Display results in a table
# print(df_results.pivot_table(index='Dataset Size', columns='Classifier', values='Accuracy'))

# # Define the module labels
# module_labels = ['MMATTER', 'MLBT']

# # Save confusion matrices
# for index, row in df_results.iterrows():
#     clf_name = row['Classifier']
#     dataset_size = row['Dataset Size']
#     cm = row['Confusion Matrix']
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap='Oranges') #plt.cm.Blue

#     # Annotate each cell with the value
#     for i in range(len(module_labels)):
#         for j in range(len(module_labels)):
#             plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

#     plt.title(f'Confusion Matrix - {clf_name} - {dataset_size} samples')
#     plt.colorbar()
#     # Set tick labels
#     plt.xticks(np.arange(len(module_labels)), module_labels)
#     plt.yticks(np.arange(len(module_labels)), module_labels)

#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     # Remove tick marks
#     plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
#     plt.savefig(f'confusion_matrix_{clf_name}_{dataset_size}.png', dpi=300)
#     plt.show()


# implementing k-fold for the model

# In[ ]:


# loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, KFold


# In[ ]:


# loading dataset by size and getting just the first column

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


# In[ ]:


# defining dataset sizes and classifiers

# Sizes of datasets
# dataset_sizes = [1000]
# dataset_sizes = [1000, 10000]
dataset_sizes = [1000, 10000, 100000, 1000000]


# Classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'LinearSVC': LinearSVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42)
}


# In[ ]:


# Function to train and evaluate classifiers 
def train_and_evaluate_classifier(model, x_train, y_train, x_test, y_test):
    # Assuming x and y are defined
    # x should be a 2D array (e.g., (1000, 32*32))
    # y should be a 2D array with three columns (e.g., (1000, 3))

    # Flatten the 32x32 images to 1D arrays for LogisticRegression, DecisionTreeClassifier, LinearSVM, KNN, RandomForests
    x_train_flatten = x_train.reshape(x_train.shape[0], -1)
    x_test_flatten = x_test.reshape(x_test.shape[0], -1)
    model.fit(x_train_flatten, y_train)
    y_pred = model.predict(x_test_flatten)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm


# In[ ]:


# Function to train and evaluate classifiers using k-fold cross-validation and giving confusion matrix and accuracy as results
def train_and_evaluate_classifier_kfold(model, x, y, k_fold=5):
    x_flatten = x.reshape(x.shape[0], -1)
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    
    scores = []
    confusion_matrices = []

    for train_index, test_index in kf.split(x_flatten):
        x_train, x_test = x_flatten[train_index], x_flatten[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        scores.append(accuracy)
        confusion_matrices.append(cm)

    return scores, confusion_matrices


# In[ ]:


# # Results storage
# results = []

# # Loop through different dataset sizes
# for size in dataset_sizes:
#     # Generate dataset
#     x, y = get_dataset(size)
    
#     # Split dataset
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
#     # Loop through classifiers
#     for clf_name, clf in classifiers.items():
#         # Train and evaluate classifier
#         accuracy, cm = train_and_evaluate_classifier(clf, x_train, y_train, x_test, y_test)
#         results.append({
#             'Dataset Size': size,
#             'Classifier': clf_name, 
#             'Accuracy': accuracy, 
#             'Confusion Matrix': cm})
# # Create a DataFrame from results
# df_results = pd.DataFrame(results)
# # Save the DataFrame to a text file
# df_results.to_csv('binary_classification_results.txt', index=False, sep='\t')


# In[ ]:


# Results storage
results_kfold = []
results_kfold_errorbar=[]
# Loop through different dataset sizes
for size in dataset_sizes:
    # Generate dataset
    x, y = get_dataset(size)
    
    # Loop through classifiers
    for clf_name, clf in classifiers.items():
        # Evaluate classifier using k-fold cross-validation
        fold_scores, fold_conf_matrices = train_and_evaluate_classifier_kfold(clf, x, y)
        
        # Store results for each fold
        for fold_num, (score, cm) in enumerate(zip(fold_scores, fold_conf_matrices), start=1):
            results_kfold.append({
                'Dataset Size': size,
                'Classifier': clf_name,
                'Fold Number': fold_num,
                'Accuracy': score,
                'Confusion Matrix': cm
            })
        # Calculate mean and standard deviation of accuracy scores
        mean_accuracy = np.mean(fold_scores)
        std_accuracy = np.std(fold_scores)
        
        # Store results
        results_kfold_errorbar.append({
            'Dataset_Size': size,
            'Classifier': clf_name,
            'Mean_Accuracy': mean_accuracy,
            'Std_Accuracy': std_accuracy
        })
# Create a DataFrame from k-fold results
df_results_kfold = pd.DataFrame(results_kfold)
# Save the DataFrame to a text file
df_results_kfold.to_csv('binary_classification_results_kfold.txt', index=False, sep='\t')
# Display results in a table
print(df_results_kfold)

# Create a DataFrame from k-fold results
df_results_kfold_errorbar = pd.DataFrame(results_kfold_errorbar)
# Save the DataFrame to a text file
df_results_kfold_errorbar.to_csv('binary_classification_results_kfold_errorbar.txt', index=False, sep='\t')
# Display results in a table
print(df_results_kfold_errorbar)


# In[ ]:


# Define the module labels
module_labels = ['MMATTER', 'MLBT']


# In[ ]:


# # Save confusion matrices
# for index, row in df_results.iterrows():
#     clf_name = row['Classifier']
#     dataset_size = row['Dataset Size']
#     cm = row['Confusion Matrix']
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap='Oranges') #plt.cm.Blue

#     # Annotate each cell with the value
#     for i in range(len(module_labels)):
#         for j in range(len(module_labels)):
#             plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

#     plt.title(f'Confusion Matrix - {clf_name} - {dataset_size} samples')
#     plt.colorbar()
#     # Set tick labels
#     plt.xticks(np.arange(len(module_labels)), module_labels)
#     plt.yticks(np.arange(len(module_labels)), module_labels)

#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     # Remove tick marks
#     plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
#     plt.savefig(f'confusion_matrix_{clf_name}_{dataset_size}.png', dpi=300)
#     plt.show()


# In[ ]:


# Save confusion matrices for each fold
for index, row in df_results_kfold.iterrows():
    clf_name = row['Classifier']
    dataset_size = row['Dataset Size']
    fold_num = row['Fold Number']
    cm = row['Confusion Matrix']

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Oranges') # plt.cm.Blues

    # Annotate each cell with the value
    for i in range(len(module_labels)):
        for j in range(len(module_labels)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    plt.title(f'Confusion Matrix - {clf_name} - {dataset_size} samples - Fold {fold_num}')
    plt.colorbar()
    # Set tick labels
    plt.xticks(np.arange(len(module_labels)), module_labels)
    plt.yticks(np.arange(len(module_labels)), module_labels)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Remove tick marks
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    plt.savefig(f'confusion_matrix_{clf_name}_{dataset_size}_fold_{fold_num}.png', dpi=300)
    plt.show()


# In[ ]:


# # Plotting with different markers for each classifier
# plt.figure(figsize=(12, 8))
# markers = ['o', 's', '^', 'D', 'v']  # You can customize the markers here

# for clf_name, group, marker in zip(classifiers.keys(), df_results.groupby('Classifier'), markers):
#     plt.plot(group[1]['Dataset Size'], group[1]['Accuracy'], label=clf_name, marker=marker)


# plt.title('Binary Classification Accuracy for Different Dataset Sizes')
# plt.xlabel('Dataset Size')
# plt.xscale('log')
# plt.ylabel('Accuracy')
# plt.legend()

# # Save the plot with high resolution (300 dpi)
# plt.savefig('binary_classification_accuracy_plot.png', dpi=300)
# plt.show()

# # Display results in a table
# print(df_results.pivot_table(index='Dataset Size', columns='Classifier', values='Accuracy'))


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the DataFrame from the saved file
#df_results = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/binary_classification_results_kfold_errorbar.txt", sep='\t')
df_results= df_results_kfold_errorbar
print(df_results)
# Set a seaborn style (optional)
sns.set(style="whitegrid")


# Define a dictionary to map classifiers to markers
marker_dict = {
    'Logistic Regression': 'o',
    'Decision Tree': 's',
    'LinearSVC': '^',
    'KNN': 'v',
    'Random Forest': 'D'
}
plt.figure(figsize=(10, 6))

for clf_name, group in df_results.groupby('Classifier'):
    plt.errorbar(
        group['Dataset_Size'],
        group['Mean_Accuracy'],
        yerr=group['Std_Accuracy'],
        label=clf_name,
        marker=marker_dict.get(clf_name, 'o'),  # Use 'o' as default marker if not found in the dictionary
        capsize=5
    )

plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel('Dataset Size (log scale)')
plt.ylabel('Mean Accuracy')
plt.title('Binary Classification Accuracy with Error Bars for Different Dataset Sizes')
plt.legend()
# plt.grid(True)
# Save the plot with high resolution (300 dpi)
plt.savefig('binary_classification_accuracy_errorbar_plot.png', dpi=300)
plt.show()

