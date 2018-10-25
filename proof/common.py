import pandas as pd
import os

script_dir = os.path.dirname(__file__)
train_path = os.path.join(script_dir,'./data/ml-prove/train.csv')
val_path = os.path.join(script_dir,'./data/ml-prove/validation.csv')
test_path = os.path.join(script_dir,'./data/ml-prove/test.csv')

num_classes = 6

def load_dataset_multi_classes(path):
    df = pd.read_csv(path,header = None)
    num_cols = len(df.columns)
    names = []
    for i in range(num_cols):
        if i < num_cols - 6:
            names.append('X'+str(i))
        if i >= num_cols - 6:
            j = (i - num_cols + 6)
            names.append('Y'+str(i))
            df[i] = df[i].map({-1:0,1:1})
    df.columns = names
    return df


def load_dataset(path):
    df = pd.read_csv(path,header = None)
    num_cols = len(df.columns)
    drop_indices = []
    for i in range(5):
        drop_indices.append(df.columns[num_cols - 6+i])
    df.drop(drop_indices, axis=1, inplace=True)

    num_cols = len(df.columns)
    names = []
    for i in range(num_cols-1):
        names.append('X'+str(i+1))
    names.append('Y')
    df.columns = names
    df['Y'] = df['Y'].map({-1:0, 1:1})
    return df

def get_train_val_test_multi_classes():
    train = load_dataset_multi_classes(train_path)
    val = load_dataset_multi_classes(val_path)
    test = load_dataset_multi_classes(test_path)
    return train, val, test

def get_train_val_test():
    train = load_dataset(train_path)
    val = load_dataset(val_path)
    test = load_dataset(test_path)
    return train,val,test

def get_X_Y(data):
    x_cols = [c for c in data.columns if c[0] == 'X']
    y_cols = [c for c in data.columns if c[0] == 'Y']
    X = data[x_cols]
    Y = data[y_cols]
    if len(y_cols) == 1:
        Y = pd.get_dummies(Y[y_cols[0]], prefix="Y")
    return X,Y

def onehot_to_bin(arr):
    if isinstance(arr, pd.DataFrame):
        arr = arr.as_matrix()
    return [a[1] > a[0] for a in arr]