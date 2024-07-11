#!/usr/bin/env python
# coding: utf-8

# In[8]:


print("Hello world")
2+2


# In[6]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[ ]:


# import tensorflow as tf
# gpu_available = tf.test.is_gpu_available()

# print (f"gpu_available={0}",gpu_available)
# is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
# print (f"is_cuda_gpu_available={0}",is_cuda_gpu_available)
# is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))
# print (f"is_cuda_gpu_min_3={0}",is_cuda_gpu_min_3)


# In[ ]:


# import tensorflow as tf
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

