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
    
load_namespace()


# In[2]:


# Loading/Preparing Environment for simulation
from jet_ml.config import Config
folds=5
epochs=2
dataset_size=1000 #10800000 #1000000
model_name="res_net"
simulation_name=f"alpha_s_{model_name}_{folds}_fold_{epochs}_epoch_{int(dataset_size/1000)}k_dataset_size"

config=Config(simulation_name=simulation_name)
config.__str__()


# In[3]:


from IPython.display import display
# with tf.device("CPU"):
import jet_ml.classifiers.alpha_s.preprocess_dataset as pred
(x,y_raw,y_df)=pred.preprocess_dataset_for_alpha_s(dataset_size)
y_classes=y_df.columns
y=y_df.values
display("y_classes: ",y_classes)
display("y: ",y[:5])
display("y_raw: ",y_raw[:5])


# In[4]:


from jet_ml.classifiers.alpha_s.preprocess_dataset import preprocess_dataset_for_resnet

WIDTH = 256
HEIGHT = 256
import tensorflow as tf
with tf.device("/CPU:0"):
    x_resized,y_resized= preprocess_dataset_for_resnet(x,y,WIDTH,HEIGHT)


# In[ ]:


TRAIN_PCT = 0.9
TRAIN_CUT = int(len(x) * TRAIN_PCT)

x_df_train_cut = x_resized[0:TRAIN_CUT]
x_df_validate_cut = x_resized[TRAIN_CUT:]

y_df_train_cut = y_resized[0:TRAIN_CUT]
y_df_validate_cut = y_resized[TRAIN_CUT:]


print(f"Training size: {len(x_df_train_cut)}")
print(f"Validate size: {len(x_df_validate_cut)}")
from jet_ml.dataset import create_train_data_generator, create_validation_data_generator
# Usage
train_generator = create_train_data_generator(x_df_train_cut, y_df_train_cut)
val_generator = create_validation_data_generator(x_df_validate_cut, y_df_validate_cut)
display("train_generator_x: ",train_generator.__next__()[0].shape)
display("train_generator_y: ",train_generator.__next__()[1].shape)
display("val_generator_x: ",val_generator.__next__()[0].shape)
display("val_generator_y: ",val_generator.__next__()[1].shape)
from jet_ml.models.resnet import build_model

with tf.device('/GPU:0'):#/GPU:0
    import tensorflow as tf
    # Enable logging of device placement
    tf.debugging.set_log_device_placement(True)
    
    from tensorflow.keras.layers import Input
    input_tensor = Input(shape=(HEIGHT, WIDTH, 3))
    model=build_model(input_tensor,num_classes=3,activation='softmax')
    # model.summary()
    import tensorflow as tf

    from jet_ml.models.helpers import compile_model
    model=compile_model(model)
# model.summary()

from jet_ml.models.resnet import train_model
epochs=2
monitor='val_loss' #'val_accuracy' or 'val_loss'
import tensorflow as tf
with tf.device('/GPU:0'):#/GPU:0
    train_model(model,train_generator,val_generator,epochs=epochs,monitor=monitor)


# In[ ]:


from jet_ml.evaluation import get_accuracy
predictions, accuracy = get_accuracy(model=model, data_generator=val_generator)  # or use train_generator

print("Predicted classes:", predictions)
print("Accuracy:", accuracy)


# In[5]:


from jet_ml.dataset import create_train_data_generator, create_validation_data_generator
from jet_ml.models.resnet import build_model
from jet_ml.models.helpers import compile_model
from jet_ml.models.resnet import train_model
from jet_ml.evaluation import get_accuracy

# fold, shuffle, x, y_raw
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
k_fold=StratifiedKFold(folds,shuffle=False)

out_of_sample_y=[]
out_of_sample_pred=[]
fold=0
folds_accuracy=[]
epochs_needed = []
times_taken=[]
#Must specify y StratifiedKFold for classification
for train,test in k_fold.split(x,y_raw):
    fold+=1
    print(f"Fold #{fold}")

    x_train=x_resized[train]
    y_train=y_resized[train]
    train_generator = create_train_data_generator(x_train, y_train)
    # Get a batch of data
    x_batch, y_batch = next(train_generator)

    # Get the shape of y (labels) from the batch
    x_batch_shape = x_batch.shape

    print("Shape of y (labels) in the batch:", x_batch_shape)
    
    #log train_generator size
    # print(f"train_generator size: {len(train_generator)}")
    x_test=x_resized[test]
    y_test=y_resized[test]
    val_generator = create_validation_data_generator(x_test, y_test)

    # from jet_ml.models import resnet
    from tensorflow.keras.layers import Input
    
    input_tensor = Input(shape=(HEIGHT, WIDTH, 3))
    output_shape=y.shape[1]
    activation='softmax'
    import tensorflow as tf
    with tf.device('GPU:0'):#/GPU:0
        model=build_model(input_tensor,num_classes=output_shape,activation=activation)
        # model.summary()
        import tensorflow as tf
        from jet_ml.models.helpers import compile_model
        model=compile_model(model)

        epochs=2
        monitor='val_loss' #'val_accuracy' or 'val_loss'
        
        model, history,elapsed_time,stopped_epoch=train_model(model,
                                                            train_generator,val_generator,
                                                            epochs=epochs,monitor=monitor,
                                                            fold=fold)

    from jet_ml.evaluation import get_accuracy
    pred, score=get_accuracy(model=model, data_generator=val_generator)
    print("Accuracy:", score)
    folds_accuracy.append(score)
    times_taken.append(elapsed_time)    
    epochs_needed.append(epochs)

    out_of_sample_y.append(y_test)
    out_of_sample_pred.append(pred)
    print(f"Fold score (accuracy): {score}")

    from jet_ml.evaluation import save_training_history
    save_training_history(history=history,fold=fold)

    from jet_ml.evaluation import plot_training_history
    plot_training_history(history=history,fold=fold)

from jet_ml.evaluation import save_training_stats
save_training_stats(accuracies=folds_accuracy,
                    epochs_needed=epochs_needed,
                    times_taken=times_taken)

# Build the oos prediction list and calculate the error.
out_of_sample_y=np.concatenate(out_of_sample_y)
display("out_of_sample_y.shape: ", out_of_sample_y.shape)
display("out_of_sample_y: ",out_of_sample_y[:5])

out_of_sample_pred=np.concatenate(out_of_sample_pred)

# convert the out_of_sample_y to a 1D array
out_of_sample_y_compare=np.argmax(out_of_sample_y,axis=1)# For accuracy and confusion matrix calculation
display("out_of_sample_compare.shape: ",out_of_sample_y_compare.shape)
display("out_of_sample_compare" ,out_of_sample_y_compare[:5])


# In[6]:


from jet_ml.evaluation import calculate_accuracy
calculate_accuracy(out_of_sample_y_compare,out_of_sample_pred)


# In[8]:


from jet_ml.evaluation import store_out_of_sample_y_and_predictions
# convert out_of_sample_y from (1000, 3, 1) to (1000, 3)
out_of_sample_y_squeezed=np.squeeze(out_of_sample_y,axis=2)
store_out_of_sample_y_and_predictions(y_df,out_of_sample_y_squeezed,out_of_sample_pred,y_classes)


# In[9]:


from jet_ml.evaluation import calculate_confusion_matrix
calculate_confusion_matrix(out_of_sample_y_compare, out_of_sample_pred,y_classes)

