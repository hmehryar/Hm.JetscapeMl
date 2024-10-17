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


#implementing the preprocess_dataset_for_resnet function
def preprocess_dataset_for_resnet(x,y,width=32,height=32):
    display("x.shape {0}".format(x.shape))
    display("y.shape {0}".format(y.shape))
    
    from jet_ml.dataset import is_normalized,convert_to_rgb,resize_images,resize_y
    display("Data is normalized: {0}".format(is_normalized(x)))

    x_rgb=convert_to_rgb(x)
    display("x_rgb.shape {0}".format(x_rgb.shape))
    
    x_resized=resize_images(x_rgb,width,height)
    display("x_resized.shape {0}".format(x_resized.shape))
    
    y_resized=resize_y(y)
    display("y_resized.shape {0}".format(y_resized.shape))
    return x_resized,y_resized

def preprocess_dataset_for_pointnet(x,y):
    display("x.shape {0}".format(x.shape))
    display("y.shape {0}".format(y.shape))
    
    from jet_ml.dataset import is_normalized
    display("Data is normalized: {0}".format(is_normalized(x)))
    from jet_ml.models.pointnet import get_dataset_points
    x_points = get_dataset_points(x)
    
    print("x_points:",type(x_points), x_points.size, x_points.shape)
    print("y:",type(y), y.size,y.shape)

    return x_points,y