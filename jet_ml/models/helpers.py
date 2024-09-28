from keras.callbacks import EarlyStopping
# Extract stopped_epoch from the EarlyStopping callback
def extract_stopped_epoch(callbacks):
    stopped_epoch = None
    for callback in callbacks:
        if isinstance(callback, EarlyStopping):
            stopped_epoch = callback.stopped_epoch
            return stopped_epoch

from jet_ml.config import Config
import os.path as path
def get_best_model_filename(model_name,fold=None):
    best_model_filename=model_name
    if fold!=None:
       best_model_filename=f"{best_model_filename}_fold_{fold}"
    best_model_filename=f"{best_model_filename}_model.keras"
    best_model_filename=path.join(Config().SIMULATION_MODELS_DIR, best_model_filename)
    return best_model_filename

import tensorflow as tf
def compile_model(model):
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
    return model

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os.path as path
import time
def get_callbacks( monitor='val_loss',
                  #early_stopping_patience=10, #for test
                  early_stopping_patience=100, #for run
                  reduce_lr_patience=5,reduce_lr_factor=0.2,reduce_lr_min_lr=0.001,
                  model_checkpoint_best_model_filename=None,
                  model_checkpoint_save_best_only=True,
                  verbose=1):
    es=None
    if isinstance(monitor, str):
        mode = None
        if 'loss' in monitor:
            mode = 'min'
        elif 'accuracy' in monitor:
            mode = 'max'
        assert mode != None, 'Check the monitor parameter!'

        es = EarlyStopping(monitor=monitor, min_delta=1e-3, patience=early_stopping_patience,
                        mode='auto',restore_best_weights=True,
                        verbose=verbose)
        mcp = ModelCheckpoint(model_checkpoint_best_model_filename,monitor=monitor, 
                        save_best_only=model_checkpoint_save_best_only, mode=mode,
                        verbose=verbose)
        
        rlp = ReduceLROnPlateau(monitor=monitor, mode=mode, 
                                factor=reduce_lr_factor, patience=reduce_lr_patience,
                                min_lr=reduce_lr_min_lr, 
                                verbose=1)
        return [es, rlp, mcp]
    else:
        raise Exception("The monitor is not an string,\
                        the non-string section is not implmented for this code")
    
