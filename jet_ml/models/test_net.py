from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,MaxPooling2D

def build_model(input_shape,num_classes,activation='softmax'):
    model=Sequential(name="testnet")
    model.add(Conv2D(32,kernel_size=(3,3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64,kernel_size=(3,3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation=activation))
    return model

import keras

def compile_model(model):
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
    return model

from keras.callbacks import EarlyStopping, ModelCheckpoint
import os.path as path
def get_callbacks( best_model_filename,monitor='val_loss'):
    
    es=None
    if isinstance(monitor, str):
        mode = None
        if 'loss' in monitor:
            mode = 'min'
        elif 'accuracy' in monitor:
            mode = 'max'
        assert mode != None, 'Check the monitor parameter!'

    # patience=25 for run
    # patience=10 for test
        es = EarlyStopping(monitor=monitor, min_delta=1e-3, patience=100, 
                            verbose=1, mode='auto', restore_best_weights=True)
        mcp = ModelCheckpoint(best_model_filename, monitor=monitor, 
                          save_best_only=True, mode=mode, verbose=1)
        return [es, mcp]    
    else:
        #the monistor should be an early stopping object
        es = monitor
        return [es]
    
    
    


import time
from jet_ml.config import Config
def train_model(model,x_train,y_train, x_test,y_test, epochs, batch_size, monitor,fold=None):
    keras.backend.clear_session()
    from jet_ml.models.helpers import get_best_model_filename
    best_model_filename=get_best_model_filename(model.name,fold=fold)
    
    callbacks = get_callbacks( best_model_filename,monitor=monitor)
    start_time=time.time()

    history=model.fit(x_train,y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test,y_test),
            callbacks=callbacks,
            )
    from jet_ml.models.helpers import extract_stopped_epoch
    stoppped_epoch=extract_stopped_epoch(callbacks=callbacks)

    elapsed_time=time.time()-start_time
    import jet_ml.helpers as helpers
    print("Elpased time: {}".format(helpers.hms_string(elapsed_time)))
    return model, history, elapsed_time,stoppped_epoch