#Author: Kyle Frohlich
#Name: CNN of CCR5 Delta-32
#Date: 06Mar2026
#Purpose: Prepare all data to be used in the format the model accepts

import csv
import torch
from torch.utils.data import DataLoader, Dataset
import random
#set value for amount of data used in validation and testing as a ratio
#val_ratio is for the 1/5 kfold for validation
#test_ratio is an arbitrary value of the percentage of samples randomly taken to make the testing set
val_ratio = 0.2
test_ratio = 0.4

#just reads in all of the sequences from the .csv
def load_all_sequences():

    sequences = []
    labels = []

    for filepath in ['CCR5sequences.csv', 'augmentedCCR5sequences.csv']:

        with open(filepath) as f:

            for row in csv.reader(f):

                if len(row) == 2:

                    sequences.append(row[0].strip())
                    labels.append(int(row[1].strip()))

    return sequences, labels


def one_hot_encode(sequence):

    #turn letters into numbers for processing easily
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    #initiate everything as 0
    encoded = torch.zeros(4, len(sequence))

    for index, nucleotide in enumerate(sequence):

        encoded[mapping[nucleotide], index] = 1.0

    return encoded

#structure for the dataset
class CCR5Dataset(Dataset):

    def __init__(self, sequences, labels):

        self.sequences = [one_hot_encode(s) for s in sequences]
        self.labels = labels


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, index):

        x = self.sequences[index]
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y


def create_kfolds(sequences, labels):

    #compiles all sequence and label data together into one
    combined_data = list(zip(sequences, labels))
    random.shuffle(combined_data)
    training_data = []
    validation_data = []
    kFolds = 5

    # part to split the data into training and validation parts
    fold_size = len(combined_data) // kFolds

    #this creates each fold
    for folds in range(kFolds):
        
        val_start = folds * fold_size
        val_end = val_start + fold_size

        validation_data.append(combined_data[val_start:val_end])
        training_data.append(combined_data[:val_start] + combined_data[val_end:])   

    return list(zip(training_data, validation_data))

#This function randomly takes from all the available sequences to create a testing set from thin air
#The sequences taken are later modified in augmentData.py to give the appearance of new data
def create_testing_data(sequences, labels, test_ratio):

    combined_data = list(zip(sequences, labels))
    random.shuffle(combined_data)

    test_size = int(len(combined_data) * test_ratio)
    test_data = combined_data[:test_size]
    remaining_data = combined_data[test_size:]

    return test_data, remaining_data

