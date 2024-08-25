import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def conv2d_layer_block(prev_layer, filters, dropout_rate, input_shape=None):
    if input_shape != None:
        prev_layer.add(Conv2D(filters=filters, kernel_size=5,
                              kernel_initializer='he_uniform',
                              padding='same',
                              activation='relu',
                              kernel_regularizer=l2(l2=0.02),
                              input_shape=input_shape
                             )
                      )
    else:
        prev_layer.add(Conv2D(filters=filters, kernel_size=5,
                              kernel_initializer='he_uniform',
                              padding='same',
                              activation='relu',
                              kernel_regularizer=l2(l2=0.02),
                             )
                      )
    prev_layer.add(Conv2D(filters=filters, kernel_size=5,
                              kernel_initializer='he_uniform',
                              padding='same',
                              activation='relu',
                              kernel_regularizer=l2(l2=0.02)
                             )
                      )    
    prev_layer.add(MaxPool2D(pool_size=(2, 2)))
    prev_layer.add(Dropout(dropout_rate))
    
    return prev_layer

def fc_layer_block(prev_layer, units, dropout_rate, num_classes=0,activation='softmax'):
    if num_classes==0:
        prev_layer.add(Dense(units, activation='relu',
                             kernel_initializer='he_uniform',
                             kernel_regularizer=l2(l2=0.02)
                            )
                      )
        prev_layer.add(Dropout(dropout_rate))
    else:
        prev_layer.add(Dense(num_classes, activation=activation))

    return prev_layer

def build_model(input_shape, num_classes=3,activation='softmax', dropout1=0.2, dropout2= 0.2):
    model = Sequential(name="vgg16_net")
    model = conv2d_layer_block(model, 256, dropout1, input_shape)
    model = conv2d_layer_block(model, 256, dropout1)
    model = conv2d_layer_block(model, 256, dropout1)
    model = conv2d_layer_block(model, 256, dropout1)
    #model = conv2d_layer_block(model, 128, dropout1)
    model.add(Flatten())
    model = fc_layer_block(model, 1024, dropout2)
    model = fc_layer_block(model, 1024, dropout2)
    model = fc_layer_block(model, 1024, dropout2)
    model = fc_layer_block(model, 1024, dropout2)
    model = fc_layer_block(model, 1, None, num_classes=num_classes,activation=activation)
    
    return model
# loss='categorical_crossentropy'

def compile_model(model,
                  loss=keras.losses.categorical_crossentropy,learning_rate=5e-6):
    model.compile(loss=loss,
                optimizer=Adam(learning_rate = learning_rate),
                metrics=['accuracy'])
    return model


from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os.path as path
import time

def get_callbacks( best_model_filename,monitor='val_loss'):

    es=None
    if isinstance(monitor, str):
        mode = None
        if 'loss' in monitor:
            mode = 'min'
        elif 'accuracy' in monitor:
            mode = 'max'
        assert mode != None, 'Check the monitor parameter!'

        es = EarlyStopping(monitor=monitor, min_delta=1e-3, patience=100,
                        verbose=1,mode='auto',restore_best_weights=True)
        mcp = ModelCheckpoint(best_model_filename, monitor=monitor, 
                            save_best_only=True, mode=mode, verbose=1)
        
        rlp = ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.2, patience=5,
                                min_lr=0.001, verbose=1)
        return [es, rlp, mcp]
    else:
        raise Exception("The monitor is not an string,\
                        the non-string section is not implmented for this code")
    
    


import keras
from jet_ml.config import Config
def train_model(model,x_train,y_train, x_test,y_test, epochs, batch_size, monitor,fold=None):
    keras.backend.clear_session()
    from jet_ml.models.helpers import get_best_model_filename
    best_model_filename=get_best_model_filename(model.name,fold=fold)
    
    callbacks = get_callbacks( best_model_filename,monitor=monitor)
    start_time=time.time()

    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        verbose=1, 
                        batch_size=batch_size, 
                        validation_data=(x_test,y_test),
                        callbacks=callbacks)
    from jet_ml.models.helpers import extract_stopped_epoch
    stoppped_epoch=extract_stopped_epoch(callbacks=callbacks)
    elapsed_time=time.time()-start_time
    import jet_ml.helpers as helpers
    print("Elpased time: {}".format(helpers.hms_string(elapsed_time)))
    return model, history, elapsed_time,stoppped_epoch