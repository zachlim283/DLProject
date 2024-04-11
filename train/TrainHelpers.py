import torch
import pickle
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm.notebook import tqdm
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def train_model(Autoencoder, train_dataloader, val_dataloader, val_dataset, n_input, n_hidden, dataset_ver, encoder_ver, tag, early_stop, dropout_rate, num_epochs, lr_rate):  
    # tensorboard logging
    writer = SummaryWriter(log_dir=f"tensorboard/{tag}")
    
    ### define model + optimizer ###
    autoencoder = Autoencoder(n_input, n_hidden, dropout_rate).to(device)
    print(autoencoder)
    optimizer = optim.Adam(autoencoder.parameters(),
                            lr=lr_rate,
                            betas=(0.9, 0.999),
                            eps=1e-8)
    optimizer.zero_grad()
    # early stopping (if applicable)
    early_stopper = EarlyStopper(patience=1, min_delta=10)

    best_loss = np.inf
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch+1}...')
        
        autoencoder.train()
        for X_batch, _ in tqdm(train_dataloader):
            X = X_batch.to(device)
            # print(X.shape)
            pred = autoencoder(X)
            train_loss = autoencoder.loss(pred, X)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(train_loss.item())
        
        autoencoder.eval()
        val_loss = 0.0
        for X_batch, _ in tqdm(val_dataloader):
            X = X_batch.to(device)

            pred = autoencoder(X)
            val_loss += autoencoder.loss(pred, X).item() * X_batch.size(0)        

        val_losses.append(val_loss / len(val_dataset))

        print(f'Epoch [{epoch+1}/{num_epochs}]\nTraining Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        writer.add_scalars("Loss", {"train": train_losses[-1], 
                                    "val": val_losses[-1]}, epoch)

        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]

            # Save Encoder
            torch.save(autoencoder.encoder, f'ML_Models/{dataset_ver}_{encoder_ver}_{tag}_encoder.pt')
            # Save Full Autoencoder
            torch.save(autoencoder, f'ML_Models/{dataset_ver}_{encoder_ver}_{tag}_Autoencoder.pt')
        
        if early_stop and early_stopper.early_stop(val_losses[-1]):
            print("Early Stopping...")             
            break

    writer.close()
    return train_losses, val_losses


def plot_loss_graph(train_losses, val_losses, tag):
    plt.plot(range(len(train_losses)), train_losses, label= 'Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss across Epochs')
    plt.legend()
    plt.savefig(f'images/loss_{tag}.png')
    plt.show()

    # log these losses separately
    with open(f"logs/losses_{tag}.txt", "a") as file:
        file.write(f"---- Model: {tag} ----\n")
        file.write("Train Losses:\n")
        for epoch, loss in enumerate(train_losses, start=1):
            file.write(f"Epoch {epoch}: {loss}\n")
        file.write("Validation Losses:\n")
        for epoch, loss in enumerate(val_losses, start=1):
            file.write(f"Epoch {epoch}: {loss}\n")
        file.write("\n")


def encode_data(train_dataloader, dataset_ver, tag, encoder_ver):
    print('Loading Encoder Model...')
    encoder = torch.load(f'ML_Models/{dataset_ver}_{encoder_ver}_{tag}_encoder.pt').to(device)
    encoder.eval()

    encoded_data = []
    labels = []

    print('Encoding Data...')
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            X_batch, y_batch = batch
            X = X_batch.to(device)
            y = y_batch.to(device).reshape(-1, 1)

            encoded_batch = encoder(X)
            
            encoded_data.append(encoded_batch)
            labels.append(y)

        encoded_data = torch.cat(encoded_data, 0).cpu()
        labels = torch.cat(labels, 0).cpu()       

    print(f'{encoded_data.shape=}')
    print(f'{labels.shape=}')
    
    return encoded_data, labels


def train_svm(train_data, train_labels, dataset_ver, tag, encoder_ver):
    print("Training SVM Classifier...")
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.20, random_state=0)
    svm_classifier = svm.SVC(kernel='rbf', probability=True)
    svm_classifier.fit(x_train, y_train)

    test_pred = svm_classifier.predict(x_test)
    print(f'Accuracy: {accuracy_score(y_test, test_pred)}')

    # Save SVM Classifier
    with open(f'ML_Models/{dataset_ver}_{encoder_ver}_{tag}_svm_classifier.pkl', "wb") as file:
        pickle.dump(svm_classifier, file)

    return svm_classifier