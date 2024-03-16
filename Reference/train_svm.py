import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_encoder import Encoder
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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
    

def train_svm(train_data, train_labels):
    x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.20, random_state=0)

    svm_classifier = svm.SVC(kernel='rbf', probability=True)
    svm_classifier.fit(x_train, y_train)

    test_pred = svm_classifier.predict(x_test)
    print(f'Accuracy: {accuracy_score(y_test, test_pred)}')

    return svm_classifier



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
    else:
        device = torch.device("cpu")
    print(device)

    dataset_ver = '3w05s_BH'
    encoder_ver = '540h20e'

    train_dataset = TrainDataset(dataset_ver)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print('Loading Encoder Model...')
    encoder = torch.load(f'ML_Models/{dataset_ver}_{encoder_ver}_encoder.pt').to(device)
    encoder.eval()

    encoded_data = []
    labels = []

    print('Encoding Data...')
    with torch.no_grad():
        for batch in train_dataloader:
            X_batch, y_batch = batch
            X = X_batch.to(device)
            y = y_batch.to(device).reshape(-1, 1)

            encoded_batch = encoder(X)
            encoded_batch = encoded_batch.reshape(encoded_batch.shape[1], encoded_batch.shape[2])

            encoded_data.append(encoded_batch)
            labels.append(y)

        encoded_data = torch.cat(encoded_data, 0).cpu()
        labels = torch.cat(labels, 0).cpu()       
    
    print("Training SVM Classifier...")
    svm_classifier = train_svm(encoded_data, labels.ravel())

    # Save SVM Classifier
    with open(f'ML_Models/{dataset_ver}_{encoder_ver}_svm_classifier.pkl', "wb") as file:
        pickle.dump(svm_classifier, file)