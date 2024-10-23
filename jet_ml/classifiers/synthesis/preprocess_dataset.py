from IPython.display import display
def preprocess_dataset_for_synthesis(size=1000):
    import jet_ml.dataset as ds 

    (x, y_raw)=ds.load_dataset(size=size)
    x=ds.reshape_x(x)
    x=ds.normalize_x(x)

    y=preprocess_y_for_synthesis(y_raw)
    
    return x,y_raw,y

def preprocess_y_for_synthesis(y):
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical


    # Combine columns into a single string to represent the class
    y_combined = np.array(['_'.join(row) for row in y])

    # Label encode the combined classes
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_combined)

    # One-hot encode the labels
    y_one_hot = to_categorical(y_encoded)

    # Now y_one_hot is ready to be used as the target for training
    print(y_one_hot)
    return y_one_hot

