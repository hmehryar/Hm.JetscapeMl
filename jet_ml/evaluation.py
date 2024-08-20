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
def plot_roc(pred,y):
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
    
