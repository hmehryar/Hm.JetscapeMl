from IPython.display import display
def preprocess_dataset_for_alpha_s(size=1000):
    import jet_ml.dataset as ds 

    (x, alpha_s)=ds.load_dataset(size=size,working_column=1)
    x=ds.reshape_x(x)
    x=ds.normalize_x(x)
    # (y,classes)=ds.categorize_y(alpha_s)
    # num_classes=classes.size
    # return x,y,num_classes
    (y_raw,y)=ds.categorize_y(alpha_s)
    
    return x,y_raw,y

def get_preprocess_dataset_info(x,y):
    import pandas as pd
    print('x shape:',x.shape)
    print('x samples:{}'.format(x[0].shape))
    # display(pd.DataFrame(x[0,:,:,0]))
    display(pd.DataFrame(y[:10]))
    
