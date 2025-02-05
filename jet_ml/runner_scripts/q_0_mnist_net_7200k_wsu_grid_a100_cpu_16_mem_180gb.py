#!/usr/bin/env python
# coding: utf-8

# In[1]:


def load_namespace():
    import sys
    sys.path.insert(1,f'/wsu/home/gy/gy40/gy4065/hm_jetscapeml_source')#WSU Grid
    sys.path.insert(1,'/content/drive/My Drive/Projects/110_JetscapeMl/hm_jetscapeml_source')#Colab GDrive v1
    sys.path.insert(1,'/content/drive/MyDrive/Projects/110_JetscapeMl/hm_jetscapeml_source')#Colab GDrive v2
    sys.path.insert(1,f'/mnt/g/My Drive/Projects/110_JetscapeMl/hm_jetscapeml_source')#wsl gdrive
    sys.path.insert(1,'G:\\My Drive\\Projects\\110_JetscapeMl\\hm_jetscapeml_source') #Windows GDrive
    sys.path.insert(1,'/home/arsalan/Projects/110_JetscapeML/hm_jetscapeml_source/') #office tower
    sys.path.insert(1,'/home/arsalan/wsu-grid/hm_jetscapeml_source') #WSU Grid fssh
    
load_namespace()


# In[2]:


print ("Dataset Preprocessor")
from jet_ml.config import Config
print(Config())


# In[3]:


dataset_size=1000
# dataset_size=7200000
dataset_file_name=f"/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_{dataset_size}_balanced_unshuffled/"
# dataset_directory_name = Config().DATA_DIR / dataset_directory_name
# print(dataset_directory_name)

dataset_file_name = f"{Config().DATA_DIR}{dataset_file_name}"
# / dataset_file_name
print(dataset_file_name)


# In[4]:


import tensorflow as tf
tf_dataset=tf.data.Dataset.list_files(f"{dataset_file_name}*/*",shuffle=False)
image_count = len(tf_dataset)
print(image_count)


# In[5]:


for file in tf_dataset.take(3):
    print(file.numpy())


# In[6]:


classes_name=[
 'MMAT_0.4_1',
 'MLBT_0.2_1.5',
 'MLBT_0.4_2.0',
 'MLBT_0.3_2.5',
 'MLBT_0.4_2.5',
 'MLBT_0.2_2.5',
 'MLBT_0.3_1.5',
 'MLBT_0.3_2.0',
 'MMAT_0.2_1',
 'MMAT_0.3_1',
 'MLBT_0.2_2.0',
 'MLBT_0.4_1.5']
print(classes_name)


# In[7]:


train_size = int(0.9 * image_count)
train_ds = tf_dataset.take(train_size)
print(len(train_ds))
test_ds = tf_dataset.skip(train_size)
print(len(test_ds))


# In[8]:


def get_label(file_path):
    import os
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]
def get_separated_label(file_path):
    label=get_label(file_path)# Extract the combined label (e.g., 'MLBT_0.2_1.5')
    separated_label = tf.strings.split(label, "_")  # Split the label on "_"
    return separated_label


# In[9]:


labels=get_sep_label=get_separated_label('/home/arsalan/wsu-grid/hm_jetscapeml_source/data/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled/MLBT_0.2_1.5/event_0000002.npy')
print(labels)


# In[10]:


import numpy as np
import tensorflow as tf
import io

def process_image(file_path):
    label = get_separated_label(file_path)

    def load_npy(path):
        path = path.numpy().decode('utf-8')  # Convert Tensor to string
        with open(path, "rb") as f:
            img = np.load(f)  # Load .npy file as NumPy array
        # print(f"Image Min: {img.min()}, Image Max: {img.max()}")  # Check values
        return img.astype(np.float32)  # Ensure correct data type

    img = tf.py_function(load_npy, [file_path], Tout=tf.float32)  # Use tf.py_function instead of tf.numpy_function
    img = tf.reshape(img, (32, 32))  # Ensure correct shape
    return img, label


# In[11]:


img, label = process_image(f'/home/arsalan/wsu-grid/hm_jetscapeml_source/data/jet_ml_benchmark_config_01_to_09_alpha_0.2_0.3_0.4_q0_1.5_2.0_2.5_MMAT_MLBT_size_1000_balanced_unshuffled/MLBT_0.2_1.5/event_0000002.npy')
display(img)
display(label)


# In[12]:


train_ds = train_ds.map(process_image)
test_ds = test_ds.map(process_image)


# In[13]:


for image, label in train_ds.take(1):
    print("****",image)
    print("****",label)


# In[ ]:


# import tensorflow as tf

# def find_max_value(dataset):
#     max_value = tf.constant(float('-inf'))  # Initialize to the smallest possible value
#     for x, _ in dataset:  # Assuming the dataset yields (x, y)
#         batch_max = tf.reduce_max(x)  # Find max in the current batch
#         max_value = tf.maximum(max_value, batch_max)  # Update the global max
#     return max_value

# # Find max value for train_ds and test_ds
# train_max = find_max_value(train_ds)
# test_max = find_max_value(test_ds)

# # Combine to get the overall max
# overall_max = tf.maximum(train_max, test_max)

# # Print results
# print("Max value in train_ds:", train_max.numpy())
# print("Max value in test_ds:", test_max.numpy())
# print("Overall max value:", overall_max.numpy())


# In[25]:


# Step 1: Find the maximum value across the entire dataset
max_value = 0.0
for img, label in train_ds:
    max_value = max(max_value, tf.reduce_max(img).numpy())  # Get the max value of each image

print(f"Max Value Found: {max_value}")


# In[24]:


def scale_image(img, max_value):
    # Scale the image by dividing by the max value
    return img / max_value


# In[26]:


def scale_images(img, label):
    img = scale_image(img, max_value)  # Scale the image
    return img, label


# In[27]:


# Then apply the scaling function
train_ds = train_ds.map(scale_images)


# In[28]:



for image, label in train_ds.take(1):
    print("****Image: ",image.numpy().max())
    print("****Label: ",label.numpy())

