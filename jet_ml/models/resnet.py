def build_model(input_tensor,num_classes,activation='softmax'):
    from tensorflow.keras.applications.resnet50 import ResNet50
    base_model = ResNet50(
    include_top=False, weights=None, input_tensor=input_tensor,
    input_shape=None)

    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    model=Model(inputs=base_model.input,outputs=Dense(num_classes,activation=activation)(x),name='ResNet50')
    return model

import keras
def train_model(model,train_generator, val_generator, epochs,
                monitor=None,
                fold=None,
                verbose=1):
    keras.backend.clear_session()
    from jet_ml.models.helpers import get_best_model_filename
    best_model_filename=get_best_model_filename(model.name,fold=fold)
    
    from jet_ml.models.helpers import get_callbacks
    callbacks = get_callbacks(monitor=monitor,
                              model_checkpoint_best_model_filename=best_model_filename)

    import time
    start_time=time.time()

    STEP_PER_EPOCH=train_generator.n//train_generator.batch_size 
    display(STEP_PER_EPOCH)

    # Important, calculate a valid step size for the validation dataset
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
    display(STEP_SIZE_VALID)
    

    history=model.fit(train_generator, 
                    epochs=epochs, steps_per_epoch=STEP_PER_EPOCH, 
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=callbacks,
                    verbose = verbose)

    from jet_ml.models.helpers import extract_stopped_epoch
    stoppped_epoch=extract_stopped_epoch(callbacks=callbacks)
    elapsed_time=time.time()-start_time
    import jet_ml.helpers as helpers
    print("Elpased time: {}".format(helpers.hms_string(elapsed_time)))
    return model, history, elapsed_time,stoppped_epoch