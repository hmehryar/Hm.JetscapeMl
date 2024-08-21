import pandas as pd
def store_out_of_sample_y_and_predictions(y_df,out_of_sample_y,out_of_sample_pred,y_classes):
    
    # Check the shape of your data
    print(len(out_of_sample_y[0]))  # Number of columns in the data

    # Define the column names
    # Original array as a pandas Index
    original_index = pd.Index(y_classes)

    # Generate new array with "OoS_" prefix
    columns = [f"OoS_{value}" for value in original_index]

    print(columns)

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


def calculate_accuracy(out_of_sample_y_compare,out_of_sample_pred):
    from sklearn import metrics
    import numpy as np
    score=metrics.accuracy_score(out_of_sample_y_compare,out_of_sample_pred)
    score_DF=pd.DataFrame({'accuracy':[score]})
    display(score)
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
    
