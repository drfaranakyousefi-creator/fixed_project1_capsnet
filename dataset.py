import pandas as pd
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import math

# Function to filter important signals from the dataset
def filter_noisy_data(x, dataset_name): 
    item_id = {
        'metavision': [  
            220045,  # Heart Rate
            220210,  # Respiratory Rate
            220179, 220180,  # Non-invasive BP Mean
            220052,  # Arterial BP Mean
            220277   # SpO2 (target for prediction)
        ],
        'carevue': [
            211,   # Heart Rate
            618,   # Respiratory Rate
            52,    # Arterial BP Mean
            456,   # NBP Mean
            676, 678,  # Temperature
            646    # SpO2 (target for prediction)
        ]
    }

    # Keep only rows with selected item IDs
    filtered_df = x[x['itemid'].isin(item_id[dataset_name])].copy()
    return filtered_df


# Function to extract data for each patient
def extract_data_from_person(dataframe, window_lengh, dataset_name, target): 
    sparse_data = [] 
    dense_data = [] 
    label = []
    
    # Set N based on dataset type
    if dataset_name == 'metavision':  
        N = 4 
    else: 
        N = 5

    # Initialize windows
    W_sparse = [torch.zeros(N) for i in range(window_lengh)]
    W_dense = [[0 for i in range(window_lengh)] for i in range(N)]

    # Loop through all rows of this patient’s data
    for index, row in dataframe.iterrows():
        item_id = row['itemid']
        value = row['value']

        # Skip invalid values
        try:
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                continue
        except (ValueError, TypeError):
            continue

        # Data order info for each dataset
        # metavision: (HR, RR, NIBP Mean, ABP Mean)
        # carevue: (HR, RR, ABP Mean, NBP Mean, Temp)

        # Handle SpO2
        if (item_id == 646) or (item_id == 220277):  
            if target == 'spO2':   
                sparse_data.append(torch.stack(W_sparse, dim=0))
                dense_data.append(torch.tensor(W_dense).T)
                label.append(torch.tensor(value))
            elif target == 'BP':  
                a = torch.zeros(N)
                a[0] = torch.tensor(value)
                W_sparse.append(a)
                W_sparse.pop(0)
                W_dense[0].append(value)
                W_dense[0].pop(0)
            elif target == 'RR':  
                a = torch.zeros(N)
                a[1] = torch.tensor(value)
                W_sparse.append(a)
                W_sparse.pop(0) 
                W_dense[1].append(value)
                W_dense[1].pop(0)   

        # Handle BP
        elif (item_id == 52) or (item_id == 220052):
            if target == 'BP': 
                sparse_data.append(torch.stack(W_sparse, dim=0))
                dense_data.append(torch.tensor(W_dense).T)
                label.append(torch.tensor(value))
            else: 
                a = torch.zeros(N)
                a[0] = torch.tensor(value)
                W_sparse.append(a)
                W_sparse.pop(0)
                W_dense[0].append(value)
                W_dense[0].pop(0)

        # Handle Respiratory Rate
        elif (item_id == 618) or (item_id == 220210):
            if target == 'RR': 
                sparse_data.append(torch.stack(W_sparse, dim=0))
                dense_data.append(torch.tensor(W_dense).T)
                label.append(torch.tensor(value))
            else: 
                a = torch.zeros(N)
                a[1] = torch.tensor(value)
                W_sparse.append(a)
                W_sparse.pop(0) 
                W_dense[1].append(value)
                W_dense[1].pop(0)      

        # Handle Heart Rate
        elif (item_id == 211) or (item_id == 220045):   
            a = torch.zeros(N)
            a[2] = torch.tensor(value)
            W_sparse.append(a)
            W_sparse.pop(0)
            W_dense[2].append(value)
            W_dense[2].pop(0)

        # Handle BP Mean (metavision)
        elif (item_id == 220179) or (item_id == 220180):
            a = torch.zeros(N)
            a[3] = torch.tensor(value)
            W_sparse.append(a)
            W_sparse.pop(0)
            W_dense[3].append(value)
            W_dense[3].pop(0)

        # Handle NBP Mean (carevue)
        elif (item_id == 456):  
            a = torch.zeros(N)
            a[3] = torch.tensor(value)
            W_sparse.append(a)
            W_sparse.pop(0)
            W_dense[3].append(value)
            W_dense[3].pop(0)  

        # Handle Temperature
        elif (item_id == 678) or (item_id == 676): 
            a = torch.zeros(N)
            a[4] = torch.tensor(value)
            W_sparse.append(a)
            W_sparse.pop(0)
            W_dense[4].append(value)
            W_dense[4].pop(0)

    # Prepare tensors if data exists
    if len(sparse_data) > 0: 
        sparse_data = torch.stack(sparse_data, dim=0)
        dense_data = torch.stack(dense_data, dim=0)
        data = torch.stack([sparse_data, dense_data], dim=1)
        label = torch.tensor(label)

        # Shuffle data
        idx = list(range(label.shape[0]))
        random.shuffle(idx)
        data = data[idx, :, :, :]  
        label = label[idx]
    else: 
        data, label = None, None

    return data, label


# Function to extract all patients’ data
def extract_data(dataset_name, df_chartevents, w, target): 
    totol_subject_ids = df_chartevents['subject_id'].unique()
    all_user_data = [] 
    all_labels = [] 

    # Loop over each subject and get their data
    for subject_id in totol_subject_ids: 
        subject_data = df_chartevents[df_chartevents['subject_id'] == subject_id]
        filtered_df = filter_noisy_data(subject_data, dataset_name)
        data, label = extract_data_from_person(filtered_df, w, dataset_name, target)

        # Keep only valid users
        if label != None: 
            all_labels.append(label)
            all_user_data.append(data)

        # Each data tensor has shape: (samples, 2, w, N)
        # N = 5 for 'carevue' and 4 for 'metavision'
    return torch.concat(all_user_data, dim=0), torch.concat(all_labels, dim=0)


# Data preparing class
class data_preparing: 
    def __init__(self, data_frame, dataset_name, w, test_size, target='spO2'):  
        # target can be 'spO2', 'BP', or 'RR'
        self.data, self.label = extract_data(dataset_name, data_frame, w, target)
        self.test_size = test_size

    # Create DataLoader for test set
    def load_test(self, batch_size): 
        start = int((1 - self.test_size) * self.label.shape[0])
        dataset = TensorDataset(self.data[start:, :, :, :], self.label[start:])
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return test_loader

    # Create DataLoader for train set
    def load_train(self, batch_size): 
        end = int((1 - self.test_size) * self.label.shape[0])
        dataset = TensorDataset(self.data[:end, :, :, :], self.label[:end])
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader
