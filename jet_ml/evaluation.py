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
    
