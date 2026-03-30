import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def DEBUG(x=None):
    out = f'[DEBUG {Path(__file__).name}]'
    if isinstance(x,pd.DataFrame):
        out = f'{x}\n{out}'
    elif x: out += f' - {x}'
    print(out)

# ================= #
# DNA DATASET CLASS #
# ================= #

class CSVDataset(Dataset):
    def __init__(self, dataset_file_path, transform=None):
        self.dataset = pd.read_csv(dataset_file_path,header=None,usecols=[1,2])
        self.set_transform(transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        dna, label = self.dataset.loc[idx]
        label = torch.tensor(label,dtype=torch.float)
        if self.__transform: dna = self.__transform(dna)
        return dna, label

    def set_transform(self, transform):
        if isinstance(transform,str): transform = eval(transform)
        self.__transform = transform

    def get_dataloader(self, batch_size, shuffle=True, drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
  
# =========================================================================== #
  
# ================= #
# UTILITY FUNCTIONS #
# ================= #

#
# TRANSFORM FUNCTIONS:
# utility encoding function for DNA DATASET class
#

# ONEHOT Encoding

def onehot_encoding(dna_line):
    options_onehot = {'A': [1,0,0,0,0],'C' :[0,1,0,0,0], 'G':[0,0,1,0,0] ,'T':[0,0,0,1,0],'N':[0,0,0,0,1]}
    onehot_data = map(lambda e: options_onehot[e], dna_line)
    onehot_data = torch.tensor(list(onehot_data),dtype=torch.float).T
    return onehot_data

# LABELING Encoding

def labeling_encoding(dna_line):
    labels = {'A':[0],'C' :[1], 'G':[2] ,'T':[3],'N':[4]}
    labeled_data = map(lambda e: labels[e], dna_line)
    labeled_data = torch.tensor(list(labeled_data),dtype=torch.float).T
    return labeled_data

# For RNN models

def labeling_encoding_RNN(dna_line):
    labels = {'A':0,'C' :1, 'G':2 ,'T':3,'N':4}
    labeled_data = map(lambda e: labels[e], dna_line)
    labeled_data = torch.tensor(list(labeled_data),dtype=torch.long)
    return labeled_data




# ================= #
# DNA DATASET CLASS #
# ================= #


class CSVDatasetFromPD(Dataset):
    def __init__(self, dataset_pd, transform=None):
        self.dataset = dataset_pd
        self.set_transform(transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        dna, label = self.dataset.loc[idx]
        label = torch.tensor(label,dtype=torch.float)
        if self.__transform: dna = self.__transform(dna)
        return dna, label

    def set_transform(self, transform):
        if isinstance(transform,str): transform = eval(transform)
        self.__transform = transform

    def get_dataloader(self, batch_size, shuffle=True, drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def createCSVDatasetTrainVal(dataset_file_path, transform=None):
        dataset = pd.read_csv(dataset_file_path,header=None,usecols=[1,2])

        ### split dataset into train and val (80/20 split) and create the new datasets
        train_size = int(0.8 * len(dataset))
        
        ## shuffle the dataset before splitting
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

        # train_data = CSVDatasetFromPD and val_data = CSVDatasetFromPD
        train_data = CSVDatasetFromPD(dataset[:train_size].reset_index(drop=True), transform=transform)
        val_data = CSVDatasetFromPD(dataset[train_size:].reset_index(drop=True), transform=transform)
        
        return train_data, val_data