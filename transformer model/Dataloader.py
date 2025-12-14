import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from numpy import *

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

## Dataloader
batch_size = 128
class spiderdataset(Dataset):
    def __init__(self,ohe, classes,seq_len,output, variance, n_samples):
        # data loading
        self.ohe = torch.from_numpy(ohe.astype(np.float32))
        self.seq_len = torch.from_numpy(seq_len.astype(np.int64))
        self.classes = torch.from_numpy(classes.astype(np.int64)) 
        self.output = torch.from_numpy(output.astype(np.float32)).reshape((-1,1))
        self.variance = torch.from_numpy(variance.astype(np.float32)).reshape((-1,1))
        self.n_samples = n_samples        
        
    def __getitem__(self,index):
        return self.ohe[index], self.classes[index], self.seq_len[index], self.output[index], self.variance[index]

    def __len__(self):    
        return self.n_samples

def make_dataset(fold=None, standardize=True):
    scaler=None

    if not fold:
        #load train data
        ohe_train = np.load('./data_generation/datasets/x_train.npy', allow_pickle=True)
        classes_train = np.argmax(ohe_train, axis=2)
        output_train = np.load('./data_generation/datasets/y_train.npy', allow_pickle=True)
        variance_train = np.load('./data_generation/datasets/s2_train.npy', allow_pickle=True)
        seq_len_train = np.load('./data_generation/datasets/len_train.npy', allow_pickle=True)
        #Load valid data
        ohe_valid = np.load('./data_generation/datasets/x_valid.npy', allow_pickle=True)
        classes_valid = np.argmax(ohe_valid, axis=2)
        output_valid = np.load('./data_generation/datasets/y_valid.npy', allow_pickle=True)
        variance_valid = np.load('./data_generation/datasets/s2_valid.npy', allow_pickle=True)
        seq_len_valid = np.load('./data_generation/datasets/len_valid.npy', allow_pickle=True)

    #Cross-Validation Datasets
    else:
        #load train data
        ohe_train = np.load(f'./data_generation/folds/x_train_fold_{fold}.npy', allow_pickle=True)
        classes_train = np.argmax(ohe_train, axis=2)
        output_train = np.load(f'./data_generation/folds/y_train_fold_{fold}.npy', allow_pickle=True)
        variance_train = np.load(f'./data_generation/folds/s2_train_fold_{fold}.npy', allow_pickle=True)
        seq_len_train = np.load(f'./data_generation/folds/len_train_fold_{fold}.npy', allow_pickle=True)
        #Load valid data
        ohe_valid = np.load(f'./data_generation/folds/x_valid_fold_{fold}.npy', allow_pickle=True)
        classes_valid = np.argmax(ohe_train, axis=2)
        output_valid = np.load(f'./data_generation/folds/y_valid_fold_{fold}.npy', allow_pickle=True)
        variance_valid = np.load(f'./data_generation/folds/s2_valid_fold_{fold}.npy', allow_pickle=True)
        seq_len_valid = np.load(f'./data_generation/folds/len_valid_fold_{fold}.npy', allow_pickle=True)

    #Load test data
    ohe_test = np.load('./data_generation/datasets/x_test.npy', allow_pickle=True)
    classes_test = np.argmax(ohe_test, axis=2)
    output_test = np.load('./data_generation/datasets/y_test.npy', allow_pickle=True)
    variance_test = np.load('./data_generation/datasets/s2_test.npy', allow_pickle=True)
    seq_len_test = np.load('./data_generation/datasets/len_test.npy', allow_pickle=True)

    # Standardize data
    if standardize:
        scaler = StandardScaler()
        scaler.fit(output_train)
        output_train = scaler.transform(output_train)
        output_valid = scaler.transform(output_valid)
        output_test = scaler.transform(output_test)

    train_dataset = spiderdataset(ohe_train,classes_train,seq_len_train,output_train,variance_train,ohe_train.shape[0])     
    valid_dataset = spiderdataset(ohe_valid,classes_valid,seq_len_valid,output_valid,variance_valid,ohe_valid.shape[0])
    test_dataset = spiderdataset(ohe_test,classes_test,seq_len_test,output_test,variance_test,ohe_test.shape[0])

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)
      
    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=False)
      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    return train_loader, valid_loader, test_loader, ohe_valid.shape[0], ohe_valid.shape[1], ohe_test.shape[0], scaler