import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_ver):
        self.train_X = np.array(np.load(f'Generated_Datasets/train_data_{dataset_ver}.npy'))
        self.train_y = np.array(np.load(f'Generated_Datasets/train_labels_{dataset_ver}.npy'))

    def __len__(self):
        return len(self.train_X)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.train_X[idx]).float()
        y = torch.from_numpy(np.asarray(self.train_y[idx])).float()
        return X, y


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.LSTM1 = nn.LSTM(input_size=input_size, hidden_size=2*hidden_size, batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, batch_first=True)
    
    def forward(self, x):
        x, (_, _) = self.LSTM1(x)
        x, (encoded_x, _) = self.LSTM2(x)
        return encoded_x


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.LSTM1 = nn.LSTM(input_size=hidden_size, hidden_size=2*hidden_size, batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=2*hidden_size, hidden_size=output_size, batch_first=True)
        
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.repeat(60, 1, 1)
        x = x.reshape(x.shape[1], 60, self.hidden_size)
        x, (_, _) = self.LSTM1(x)
        x, (_, _) = self.LSTM2(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, in_out_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_out_size, hidden_size).to(device)
        self.decoder = Decoder(hidden_size, in_out_size).to(device)

        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
    else:
        device = torch.device("cpu")
    print(device)
    
    ############################################################################
    ##########################       PARAMETERS       ##########################
    ############################################################################

    dataset_ver = '3w05s_BH'
    n_input = 10
    n_hidden = 360
    num_epochs = 20

    ############################################################################
    ############################################################################
    ############################################################################

    # Model Params
    encoder_ver = f'{n_hidden}h{num_epochs}e'

    train_dataset = TrainDataset(dataset_ver)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = Autoencoder(n_input, n_hidden).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3,
                           betas=(0.9, 0.999),
                           eps=1e-8)
    optimizer.zero_grad()

    best_loss = np.inf

    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch+1}...')
        for batch in train_dataloader:
            X_batch, y_batch = batch
            X = X_batch.to(device)
            y = y_batch.to(device).reshape(-1, 1)

            pred = model(X)
            loss_value = model.loss(pred, X)

            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss_value.item():.4f}')
        
        if loss_value.item() < best_loss:
            best_loss = loss_value.item()

            # Save Encoder
            torch.save(model.encoder, f'ML_Models/{dataset_ver}_{encoder_ver}_encoder.pt')

            # Save Full Autoencoder
            torch.save(model, f'ML_Models/{dataset_ver}_{encoder_ver}_autoenencoder.pt')
