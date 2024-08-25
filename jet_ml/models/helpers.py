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