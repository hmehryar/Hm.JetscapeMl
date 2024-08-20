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
def get_callbacks(monitor, save_dir):
    mode = None
    if 'loss' in monitor:
        mode = 'min'
    elif 'accuracy' in monitor:
        mode = 'max'
    assert mode != None, 'Check the monitor parameter!'

    es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=100, 
                        verbose=1, mode='auto', restore_best_weights=True)
    mcp = ModelCheckpoint(path.join(save_dir, 'model.keras'), monitor=monitor, 
                          save_best_only=True, mode=mode, verbose=1)
    
    return [es, mcp]


import time
from jet_ml.config import Config
def train_model(model,x_train,y_train, x_test,y_test, epochs, batch_size, monitor):
    keras.backend.clear_session()
    simulation_path=Config().SIMULATION_MODELS_DIR/model.name
    callbacks = get_callbacks(monitor, simulation_path)
    start_time=time.time()

    history=model.fit(x_train,y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test,y_test),
            callbacks=callbacks,
            )
    elapsed_time=time.time()-start_time
    import jet_ml.helpers as helpers
    print("Elpased time: {}".format(helpers.hms_string(elapsed_time)))
    return model, history