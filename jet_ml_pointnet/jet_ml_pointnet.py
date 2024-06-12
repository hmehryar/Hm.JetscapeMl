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
# Extract the first 10 images and labels
first_10_images = dataset_x[:10]
labels = dataset_y[:10]

# Plot the images with their corresponding labels
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(first_10_images[i], cmap='gray')  # Assuming images are grayscale
    axes[row, col].set_title(f'Label: {labels[i]}')
    axes[row, col].axis('off')

# Save the plot as an image file (e.g., PNG)
plt.savefig('first_10_images_plot.png')

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming dataset.x contains the images and dataset.y contains the labels
# Extract the first two images and labels
first_2_images = dataset_x[:2]
labels = dataset_y[:2]

# Plot the images with a white background
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for i in range(2):
    axes[i].imshow(first_2_images[i], cmap='gray', vmin=0, vmax=1)  # Assuming images are normalized between 0 and 1
    axes[i].set_title(f'Label: {labels[i]}')
    axes[i].axis('off')
    axes[i].set_facecolor('white')  # Set white background

# Save the plot as an image file (e.g., PNG)
plt.savefig('first_2_images_plot.png')
# Show the plot
plt.show()


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
ax.set_title(f'Label: {label}')
ax.axis('off')
ax.set_facecolor('white')  # Set white background

# Save the plot as an image file (e.g., PNG)
plt.savefig('first_image_plot.png')

# Show the plot
plt.show()


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


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming dataset.x contains the images and dataset.y contains the labels
# Extract the first image and label
first_image = dataset_x[0]
label = dataset_y[0]

# Plot the first image with a white background
fig, ax = plt.subplots(figsize=(4, 4))

ax.imshow(first_image, cmap='viridis', vmin=0, vmax=1)  # Assuming images are normalized between 0 and 1
# ax.set_title(f'Label: {label}')
ax.axis('off')
ax.set_facecolor('white')  # Set white background

# Save the plot as an image file (e.g., PNG)
plt.savefig('first_image_plot_labelless_colored.png')

# Show the plot
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming dataset.x contains the images and dataset.y contains the labels
# Extract the first image and label
first_image = dataset_x[0]
label = dataset_y[0]

# Plot the first image with a white background
fig, ax = plt.subplots(figsize=(4, 4))

ax.imshow(first_image, cmap='jet', vmin=0, vmax=1)  # Assuming images are normalized between 0 and 1
# ax.set_title(f'Label: {label}')
ax.axis('off')
ax.set_facecolor('white')  # Set white background

# Save the plot as an image file (e.g., PNG)
plt.savefig('first_image_plot_labelless_colored_jet.png')

# Show the plot
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming dataset.x contains the images and dataset.y contains the labels
# Extract the first image and label
first_image = dataset_x[0]
label = dataset_y[0]

# Plot the first image with color for non-zero values and white for zero values
fig, ax = plt.subplots(figsize=(4, 4))

# Create a masked array to set alpha=0 for zero values
masked_image = np.ma.masked_where(first_image == 0, first_image)

# Display the image with the 'jet' colormap
cax = ax.imshow(masked_image, cmap='jet', vmin=0, vmax=1)

# Set alpha=0 for zero values
cax.set_array(np.ma.masked_where(masked_image == 0, masked_image).ravel())

ax.axis('off')
ax.set_facecolor('white')  # Set white background

# Save the plot as an image file (e.g., PNG)
plt.savefig('first_image_plot_colored_jet_nonzero.png')

# Show the plot
plt.show()


# In[ ]:


# from matplotlib.pyplot import figure
# from matplotlib.ticker import LogFormatter 
# from matplotlib import pyplot as plt, colors
# from matplotlib.transforms import offset_copy

# #input
# pi=3.14159
# bin_count=32


# In[ ]:


# def convert_event_to_image(bin_count,event_item,draw_plot=False):
#     event_v = np.vstack(event_item)
#     fig, ax = plt.subplots()
#     counts, xedges, yedges, image = ax.hist2d(event_v[:,0], event_v[:,1],
#      bins=bin_count, norm=colors.LogNorm(), weights=event_v[:,2], cmap = plt.cm.jet)
 
#     if draw_plot:
#         # plt.rcParams["figure.autolayout"] = True
#         fig.colorbar(image, ax=ax)
#         ax.set_xticks(np.arange(-3, pi, 1)) 
#         ax.set_yticks(np.arange(-3, pi, 1)) 
#         print ("Saving Image: Begin")
#         file_name='hm_jetscape_ml_plot_hist_with_white_bg.png'
#         file_path=file_name
#         plt.savefig(file_path)
#         print ("Saving Image: End")
#         file_path=simulation_directory_path+file_name
#         plt.savefig(file_path)
#         plt.show()
#         plt.close()
        
#     return counts

# event_item_sample=dataset_x[0] 
# event_item_sample_image=convert_event_to_image(bin_count,event_item_sample,True)
# print(type(event_item_sample_image), event_item_sample_image.size, event_item_sample_image.shape)
# print(np.max(event_item_sample))
# print(np.max(event_item_sample_image))


# In[ ]:


# def plot_sample_events(events_matrix_items):
#   plt.rcParams['text.usetex'] = True
#   # plt.rcParams["figure.autolayout"] = True
#   # fig, axes = plt.subplots(2, 10, figsize=[70,10], dpi=200)
#   fig, axes = plt.subplots(2, 5, figsize=[30,10], dpi=200)
  
#   # fig.subplots_adjust(top=0.8)

#   # Set titles for the figure 
#   if y_class_label_items[0]=='MMAT':
#     matter_lbt_str='MATTER, In Medium'
#   else:
#     matter_lbt_str='MATTER+LBT, In Medium'
#   suptitle=r'$Config No.{0}: {1}, Q_{{0}}=  {2}, \alpha_{{s}}= {3}$  '.format(1,matter_lbt_str,0,0)
 
#   fig.suptitle(suptitle, fontsize=20, fontweight='bold')
#   plt.setp(axes.flat, xlabel=r'$x$', ylabel=r'$y$')
 
#   for i, ax in enumerate(axes.flat):
#       event_v = np.vstack(events_matrix_items[i])
      
#       counts, xedges, yedges, image = ax.hist2d(event_v[:,0], event_v[:,1],
#       bins=bin_count, 
#       norm=colors.LogNorm(),
#       weights=event_v[:,2], cmap = plt.cm.jet)
     
      
#       # plt.colorbar(image,ax=ax, cmap=plt.cm.jet)
#       ax.set_xticks(np.arange(-3, pi, 1)) 
#       ax.set_yticks(np.arange(-3, pi, 1))
#       # Set titles for the subplot
#       ax.set_title(r'Sample {}'.format(i+1))
  

#   # cols = ['Sample #{}'.format(col+1) for col in range(0, 10)]
#   # rows = ['{}(Medium)'.format(row) for row in ['MATTER', 'MATTER+LBT']]

#   # pad = 5 # in points

#   # for ax, col in zip(axes[0], cols):
#   #     ax.annotate(str(col), xy=(0.5, 1), xytext=(0, pad),
#   #                 xycoords='axes fraction', textcoords='offset points',
#   #                 size='large', ha='center', va='baseline')

#   # for ax, row in zip(axes[:,0], rows):
#   #     ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
#   #                 xycoords=ax.yaxis.label, textcoords='offset points',
#   #                 size='20', ha='right', va='center')

 
#   # fig.tight_layout()
#   # tight_layout doesn't take these labels into account. We'll need 
#   # to make some room. These numbers are are manually tweaked. 
#   # You could automatically calculate them, but it's a pain.
#   # fig.subplots_adjust(left=0.15, top=0.95)
#   # file_name="config-0"+str(configuration_number)+"-"+y_class_label_items[0]+"-simulationsize"+str(data_size)+"-partition"+str(partition_index)+"-numofevents"+str(number_of_events_per_partition)+"-q0-"+str(q_0)+"-alphas-"+str(alpha_s)+"-sample-events.png"
#   file_name="event-sample-10.jpg"
#   # file_path=simulation_directory_path+file_name
#   file_path=file_name
#   plt.savefig(file_path)

#   plt.show()
#   plt.close()

# #Plotting 20 Sample Events Phase  from shuffled dataset
# events_matrix_items=dataset_x[0:10]
# plot_sample_events(events_matrix_items)


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

# # Add a third dimension with a constant value of 0
# dataset_x_expanded = np.expand_dims(dataset_x, axis=-1)
# print("dataset_x_expanded visualization")
# print(dataset_x_expanded[0])

# # Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(dataset_x_expanded, dataset_y_binary, test_size=test_size, random_state=42)

# # Display the shapes of the training and test datasets
# print("Training set shapes - x:", x_train.shape, " y:", y_train.shape)
# print("Training set size - x:", x_train.size, " y:", y_train.size)
# print("Test set shapes - x:", x_test.shape, " y:", y_test.shape)
# print("Test set size - x:", x_test.size, " y:", y_test.size)


# In[ ]:


# !nvidia-smi


# In[ ]:


# import plotly


# In[ ]:


# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# from jet_ml_dataset_builder.jet_ml_dataset_builder_utilities import install
# install("PointNet")
# from pointnet.model import PointNet
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint

# # Assuming your dataset is loaded in variables dataset.x and dataset.y

# # Encode the categorical labels
# label_encoder = LabelEncoder()
# dataset_y[:, 0] = label_encoder.fit_transform(dataset_y[:, 0])

# # Convert labels to binary encoding (0 or 1)
# binary_labels = to_categorical(dataset.y[:, 0], num_classes=2)

# # Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(dataset.x, binary_labels, test_size=0.2, random_state=42)

# # Create a PointNet model
# model = PointNet()

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Save the best model during training
# checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

# # Train the model
# model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, callbacks=[checkpoint])

# # Evaluate the model on the test set
# accuracy = model.evaluate(x_test, y_test)[1]
# print(f'Test Accuracy: {accuracy}')

