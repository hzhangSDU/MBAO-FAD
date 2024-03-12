



import torch
import numpy
import scipy.io
from sklearn.preprocessing import MinMaxScaler


def read_dataset(path, v):
    
    # load dataset.
    data = scipy.io.loadmat(path)
    value = data['data']
    # normalized
    scaler_feature = MinMaxScaler()
    scaler_target = MinMaxScaler()
    value_feature = scaler_feature.fit_transform(numpy.double(value[:,0:value.shape[1]-1]))
    value_target = scaler_target.fit_transform(numpy.double(value[:,value.shape[1]-1:value.shape[1]]))
    
    # split it into training set and testing set
    train_x = value_feature[0:int(value.shape[0]*v), :]
    train_y = value_target[0:int(value.shape[0]*v)]
    test_x = value_feature[int(value.shape[0]*v):value.shape[0], :]
    test_y = value_target[int(value.shape[0]*v):value.shape[0]]
    
    return torch.tensor(train_x), torch.tensor(train_y), torch.tensor(test_x), torch.tensor(test_y)
