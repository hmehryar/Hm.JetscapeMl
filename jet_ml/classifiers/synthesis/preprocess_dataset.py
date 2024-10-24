from IPython.display import display
def preprocess_dataset_for_synthesis(size=1000):
    import jet_ml.dataset as ds 

    (x, y_raw)=ds.load_dataset(size=size)
    x=ds.reshape_x(x)
    x=ds.normalize_x(x)

    y_combined,y_df=preprocess_y_for_synthesis(y_raw)
    
    return x,y_combined,y_df

def preprocess_y_for_synthesis(y):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical


    # Combine columns into a single string to represent the class
    y_combined = np.array(['_'.join(row) for row in y])

    # Label encode the combined classes
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_combined)

    # One-hot encode the labels
    y_one_hot = to_categorical(y_encoded)

    # Get the class names from the label encoder (these will be the headers)
    class_names = label_encoder.classes_

    # Convert the one-hot encoded data into a DataFrame with proper headers
    y_one_hot_df = pd.DataFrame(y_one_hot, columns=class_names)

    return y_combined,y_one_hot_df

