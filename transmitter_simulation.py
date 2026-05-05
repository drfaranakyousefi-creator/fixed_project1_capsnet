import torch
from server_net import prediction_net
import json

# Class to simulate data transfer between client and server
class Transmitter:
    def __init__(self, w, server_url, dataset_name, device):
        # Number of features depends on dataset type
        if dataset_name == 'metavision':
            N = 4
        else:
            N = 5
        # Initialize server-side prediction model
        self.model = prediction_net(w, n_features_input=N, lr=0.01)
        self.device = device

    # Convert tensors to JSON for simulation of sending data
    def data_to_json(self, x, label, status):  # x and label are both tensors
        x_copy = x.detach().cpu().tolist()
        label_copy = label.detach().cpu().tolist()
        if status == 'train':
            # Include labels in training mode
            data = {
                'prediction_iput': x_copy,
                'label': label_copy,
                'status': status
            }
        elif status == 'test':
            # Do not include labels in test mode
            data = {
                'prediction_iput': x_copy,
                'label': [],
                'status': status
            }

        # Return JSON string
        return json.dumps(data)

    # Simulate sending data to server and receiving result back
    def send_data(self, x, label, status):
        # Convert client data to JSON
        data_transfer_to_server = self.data_to_json(x, label, status)
        # Server receives JSON and parses it
        server_recieves_data = json.loads(data_transfer_to_server)

        # Extract data for server processing
        combined_embedded = server_recieves_data['prediction_iput']
        label = server_recieves_data['label']
        status = server_recieves_data['status']

        # Server processes data with its model
        result = self.model(combined_embedded, label, status)

        # Simulate sending result back to client
        data_back_to_client = json.dumps(result)
        data_recive_in_client = json.loads(data_back_to_client)

        if status == 'train':  # Return gradient in training mode
            grad = data_recive_in_client['grad']
            return torch.tensor(grad).to(self.device)
        elif status == 'test':  # Return predictions in test mode
            prediction = torch.tensor(data_recive_in_client['prediction']).to(self.device)
            return prediction
