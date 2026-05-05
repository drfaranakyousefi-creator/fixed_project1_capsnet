from client_net import client_network
from dataset import data_preparing
from transmitter_simulation import Transmitter
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# High-level class to run the whole client-side training and communication
class HTTPS(nn.Module):
    def __init__(self, w, dataset_name, batch_size, server_url, target, lr) -> None:
        super().__init__()
        # Choose device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Number of features differs by dataset
        if dataset_name == 'metavision':
            self.N = 4
        else:
            self.N = 5

        # Client-side network that handles sparse + dense parts
        self.network = client_network(w, self.N, lr=lr).to(device)

        # Load chart events CSV (path is fixed here)
        chartevents_path = "/content/drive/MyDrive/split_learning/CHARTEVENTS.csv"
        df_chartevents = pd.read_csv(chartevents_path)

        # Prepare data (this builds train/test splits inside)
        self.data = data_preparing(df_chartevents, dataset_name, w, test_size=0.2, target=target)

        # Transmitter simulates sending data to the server and getting gradients/predictions
        self.transmittion = Transmitter(w, server_url, dataset_name, device)

        self.batch_size = batch_size
        # Loss functions used for monitoring and selecting autoencoders
        self.loss_fn = nn.MSELoss()
        self.L1Loss = nn.L1Loss()

    # Train for multiple epochs and collect history
    def fit(self, epochs):
        history = {
            'loss_train': [],
            'loss_test': []
        }
        for epoch in range(epochs):
            # Run one epoch of training
            self.train_one_epoch()
            # Evaluate after the epoch
            loss_train, loss_test = self.evaluate_one_epoch()
            print(f'''
            [epoch {epoch} / {epochs}    train_loss = {loss_train}    test_loss = {loss_test}]
            ''')
            # Convert losses to numbers and save to history
            loss_test = loss_test.item()
            loss_train = loss_train.item()
            history['loss_test'].append(loss_test)
            history['loss_train'].append(loss_train)

        return history

    # One training epoch: iterate over train loader and update client network
    def train_one_epoch(self):
        for x, l in self.data.load_train(batch_size=self.batch_size):
            # x -> (batch, 2, w, N)
            prediction_inp, dense_decoder_out, dense_inp = self.network(x.to(self.device))
            # Send to transmitter (server simulation). In train mode it returns grad.
            grad = self.transmittion.send_data(prediction_inp, l, status='train')
            # Use returned gradient to train client-side autoencoders + sparse net
            self.network.train_one_batch(prediction_inp, dense_decoder_out, dense_inp, grad.clone())
        return True

    # Evaluate model on train and test sets, return average MSE losses
    def evaluate_one_epoch(self):
        loss_train = 0
        number = 0
        for x, l in self.data.load_train(batch_size=self.batch_size):
            l = l.to(self.device)
            prediction_inp, _, _ = self.network(x.to(self.device))
            # In test mode transmitter returns predictions
            prediction = self.transmittion.send_data(prediction_inp, l, status='test')
            loss_train += x.shape[0] * self.loss_fn(prediction.to(self.device), l)
            number += x.shape[0]
        loss_train = loss_train / number

        loss_test = 0
        number = 0
        for x, l in self.data.load_test(batch_size=self.batch_size):
            l = l.to(self.device)
            prediction_inp, _, _ = self.network(x.to(self.device))
            prediction = self.transmittion.send_data(prediction_inp, l, status='test')
            loss_test += x.shape[0] * self.loss_fn(prediction.to(self.device), l)
            number += x.shape[0]
        loss_test = loss_test / number
        return loss_train, loss_test

    # Transfer learned knowledge from another HTTPS object to this one
    def get_knowledge(self, HTTPS_object):
        all_auto_encoders = HTTPS_object.network.MultiAutoEncoder.autoEncoders
        # For each feature, choose the best autoencoder from the source
        for i in range(self.N):
            l1Loss = []
            # compute loss of each candidate autoencoder for feature i
            for auto_endocer in all_auto_encoders:
                l1Loss.append(self.compute_autoEnccoder_loss(auto_endocer, i))
            # choose the index with smallest loss
            min_idx = torch.argmin(torch.stack(l1Loss))
            print(f'the feature {i} chooses the autocoder {min_idx}')
            # load weights from chosen autoencoder
            weights = all_auto_encoders[min_idx].state_dict()
            self.network.MultiAutoEncoder.autoEncoders[i].load_state_dict(weights)

    # Compute L1 loss of one autoencoder on the training data for a given feature index
    def compute_autoEnccoder_loss(self, auto_endocer, i):
        loss = 0
        number = 0
        # Use a fixed batch size here (256) to estimate average L1 loss
        for x, _ in self.data.load_train(batch_size=256):
            a = x.shape[0]
            # x[:, 1, :, i] -> the dense input for feature i (batch, w)
            inp = x[:, 1, :, i]
            _, decoder_out = auto_endocer(inp)
            # accumulate L1 loss scaled by batch size
            loss += a * self.L1Loss(inp, decoder_out)
            number += a
        loss = loss / number
        return loss
