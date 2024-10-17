import pandas as pd
from IPython.display import display
def store_out_of_sample_y_and_predictions(y_df,out_of_sample_y,out_of_sample_pred,y_classes):
    
    # Check the shape of your data
    # print(len(out_of_sample_y[0]))  # Number of columns in the data

    # Define the column names
    # Original array as a pandas Index
    original_index = pd.Index(y_classes)

    # Generate new array with "OoS_" prefix
    columns = [f"OoS_{value}" for value in original_index]

    # print(columns)

    # Check if the number of columns matches
    if len(out_of_sample_y[0]) != len(columns):
        raise ValueError("Number of columns in data does not match number of column names")

    out_of_sample_y=pd.DataFrame(out_of_sample_y,columns=columns)
    out_of_sample_pred=pd.DataFrame(out_of_sample_pred,columns=["OoS_Pred_Class"])

    out_of_sample_DF=pd.concat([y_df,out_of_sample_y,out_of_sample_pred],axis=1)
    import os
    from jet_ml.config import Config 
    out_of_sample_DF.to_csv(os.path.join(Config().SIMULATION_REPORTS_DIR,"out_of_sample_DF.csv"),index=False)


def get_accuracy_cpu(model,x_test,y_test):
    #Better to run the prediction on CPU, becuase it can exhust the resources
    score=model.evaluate(x_test,y_test,verbose=0)
    print('Test loss {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))

def get_accuracy_gpu(model,x_test,y_test,batch_size):
    #for GPU prediction mode better to sample the first couple of test data
    from sklearn import metrics
    import numpy as np
    small_x=x_test[1:100]
    small_y=y_test[1:100]
    small_y2=np.argmax(small_y,axis=1)
    pred=model.predict(small_x)
    pred=np.argmax(pred,axis=1)
    score=metrics.accuracy_score(small_y2,pred)
    print('Test accuracy: {}'.format(score))


def get_accuracy(model,x_test,y_test):
    from sklearn import metrics
    pred = model.predict(x_test)
    # raw probabilities to chosen class (highest probability)
    pred = np.argmax(pred,axis=1) 
    # Measure this fold's accuracy
    y_compare = np.argmax(y_test,axis=1) # For accuracy calculation
    score = metrics.accuracy_score(y_compare, pred)  
    return pred, score

import numpy as np
from sklearn import metrics

def get_accuracy_from_generator(model, data_generator):
    """
    Calculate accuracy of the model on data from a generator.

    Parameters:
    - model: The trained Keras model.
    - data_generator: A Keras generator for validation/testing.

    Returns:
    - pred: Predicted class labels.
    - score: Accuracy score.
    """
    all_predictions = []
    all_labels = []
    batch_index = 0
    # Iterate over the generator
    for x_batch, y_batch in data_generator:
        batch_index += 1
        if batch_index >= len(data_generator)+1:
            break
        print(f"batch_index: {batch_index}")
        # Get predictions from the model
        pred = model.predict(x_batch)
        pred_labels = np.argmax(pred, axis=1)
        
        # Check if y_batch is one-hot encoded
        if y_batch.ndim > 1:  # Assuming y_batch is one-hot encoded
            y_batch_labels = np.argmax(y_batch, axis=1)
        else:  # Assuming y_batch is not one-hot encoded
            y_batch_labels = y_batch
        
        all_predictions.extend(pred_labels)
        all_labels.extend(y_batch_labels)
    
    # Calculate accuracy score
    score = metrics.accuracy_score(all_labels, all_predictions)
    
    return all_predictions, score

def get_logloss(model,x_test,y_test):
    from sklearn import metrics
    pred = model.predict(x_test)
    # raw probabilities to chosen class (highest probability)
    pred = np.argmax(pred,axis=1) 
    # Measure this fold's accuracy
    y_compare = np.argmax(y_test,axis=1) # For accuracy calculation
    score = metrics.log_loss(y_compare, pred)  
    return pred, score

def calculate_accuracy(out_of_sample_y_compare,out_of_sample_pred):
    from sklearn import metrics
    import numpy as np
    score=metrics.accuracy_score(out_of_sample_y_compare,out_of_sample_pred)
    score_DF=pd.DataFrame({'accuracy':[score]})
    display(f"accuracy: {score}")
    import os
    from jet_ml.config import Config 
    score_DF.to_csv(os.path.join(Config().SIMULATION_REPORTS_DIR,"accuracy.csv"),index=False)

# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
# Plot a confusion matrix.
# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title='Confusion matrix', 
                            cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# Plot an ROC. pred - the predictions, y - the expected output.
def plot_binary_classification_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def calculate_confusion_matrix(out_of_sample_y_compare, out_of_sample_pred,y_classes):
        import os
        import numpy as np
        from sklearn import svm, datasets
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import pandas as pd
        
        from jet_ml.config import Config
        
        # Compute confusion matrix
        cm = confusion_matrix(out_of_sample_y_compare, out_of_sample_pred)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        plt.figure()
        plot_confusion_matrix(cm,y_classes)
        cm_DF=pd.DataFrame(cm,columns=y_classes)
        cm_DF.to_csv(os.path.join(Config().SIMULATION_REPORTS_DIR,"confusion_matrix.csv"),index=False)
        plt.savefig(os.path.join(Config().SIMULATION_FIGURES_DIR,"confusion_matrix.png"), dpi=300)

        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        print(cm_normalized)
        plt.figure()
        plot_confusion_matrix(cm_normalized, y_classes, 
                title='Normalized confusion matrix')

        cm_normalized_DF=pd.DataFrame(cm_normalized,columns=y_classes)
        cm_normalized_DF.to_csv(os.path.join(Config().SIMULATION_REPORTS_DIR,"confusion_matrix_normalized.csv"),index=False)
        plt.savefig(os.path.join(Config().SIMULATION_FIGURES_DIR,"confusion_matrix_normalized.png"), dpi=300)

        plt.show()

def save_training_history(history,fold=None):
    import os
    from jet_ml.config import Config
    file_name="training_history"
    if fold !=None:
        file_name=f"{file_name}_fold_{fold}"
    file_name=f"{file_name}.csv"
      
    pd.DataFrame.from_dict(history.history).to_csv(
        os.path.join(Config().SIMULATION_REPORTS_DIR,file_name)
        ,index=False)
    print(file_name)

def save_fold_accuracy(fold_accuracy):
    import os
    from jet_ml.config import Config
    file_name="fold_accuracy.csv"
      
    pd.DataFrame(fold_accuracy,columns=["fold_accuracy"]).to_csv(
        os.path.join(Config().SIMULATION_REPORTS_DIR,file_name)
        ,index=False)
    print(f"stored all folds' accuracy in {file_name}")

def plot_training_history(history, fold=None,x_tick=5):
    import os
    from jet_ml.config import Config
    """
    Plot training and validation accuracy and loss values and save the plot with high resolution.

    Parameters:
    - history (tf.keras.callbacks.History): History object containing training/validation metrics.
    - simulation_path (str): Path to save the plot.
    - x_tick (int) steps in x axis (optional: the default value is 5).

    Returns:
    - file_path (str): File path of the saved plot.
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history:
        x_axis_length=len(history.history.get('accuracy', []))+1
        plt.plot(history.history['accuracy'])
    # if 'sparse_categorical_accuracy' in history.history:
    #     x_axis_length=len(history.history.get('sparse_categorical_accuracy', []))+1
    #     plt.plot(history.history['sparse_categorical_accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
    # if 'val_sparse_categorical_accuracy' in history.history:
    #     plt.plot(history.history['val_sparse_categorical_accuracy'])
    
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    # Set ticks on the epoch axis to display only integer values
    plt.xticks(range(0, x_axis_length, x_tick))

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Set ticks on the epoch axis to display only integer values
    plt.xticks(range(0, x_axis_length, x_tick))

    # Adjust layout and show the plot
    plt.tight_layout()

    # Save the plot with high resolution (300 dpi)
    file_name = 'accuracy_loss'
    if fold!=None:
        file_name=f"{file_name}_fold_{fold}"
    file_name=f"{file_name}.png"
    file_name = os.path.join(Config().SIMULATION_FIGURES_DIR,file_name)
    plt.savefig(file_name, dpi=300)
    plt.show()
    plt.close()

    return file_name

def save_bootstraping_history(mean_benchmark,epochs_needed,times_took,splits=None):
    import os
    from jet_ml.config import Config
    # Save the plot with high resolution (300 dpi)
    file_name = 'bootstrapping_history'
    if splits!=None:
        file_name=f"{file_name}_splits_{splits}"
    file_name=f"{file_name}.csv"
    # Combine data into rows
    data = list(zip(mean_benchmark, epochs_needed, times_took))
    pd.DataFrame(data,columns=["accuracy_score","epoch_needed","time_took"]).to_csv(
        os.path.join(Config().SIMULATION_REPORTS_DIR,file_name)
        ,index=False)
    print(f"stored all splits' history in {file_name}")

def save_training_stats(accuracies,epochs_needed,times_taken):
    import os
    from jet_ml.config import Config
    # Save the plot with high resolution (300 dpi)
    file_name = 'training_stats.csv'
    
    # Combine data into rows
    data = list(zip(accuracies, epochs_needed, times_taken))
    pd.DataFrame(data,columns=["accuracy_score","epoch_needed","times_taken"]).to_csv(
        os.path.join(Config().SIMULATION_REPORTS_DIR,file_name)
        ,index=False)
    print(f"stored all splits' history in {file_name}")

    

    
