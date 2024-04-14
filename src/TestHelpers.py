import torch
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(test_dataloader, dataset_ver, tag, encoder_ver):

    # Load the saved models
    encoder = torch.load(f'models/{dataset_ver}_{encoder_ver}_{tag}_Autoencoder.pt')
    encoder.eval()
    with open(f'models/{dataset_ver}_{encoder_ver}_{tag}_svm_classifier.pkl', 'rb') as file:
        svm_classifier = pickle.load(file)

    test_encoded_data = []
    test_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            X_batch, y_batch = batch
            X = X_batch.to(device)
            y = y_batch.to(device).reshape(-1, 1)

            test_encoded_batch = encoder.encoder(X)
            
            test_encoded_data.append(test_encoded_batch)
            test_labels.append(y)

        test_encoded_data = torch.cat(test_encoded_data, 0).cpu()
        test_labels = torch.cat(test_labels, 0).cpu()   

    print(f'{test_encoded_data.shape=}')
    print(f'{test_labels.shape=}')

    test_predictions = svm_classifier.predict(test_encoded_data)
    print(f'{test_predictions.shape=}')
    return test_labels, test_predictions


def calculate_accuracy(test_labels, test_predictions, tag):
    # Calculate accuracy score
    accuracy = accuracy_score(test_labels, test_predictions)
    accuracy_percentage = accuracy * 100
    print(f"Accuracy: {accuracy_percentage:.2f}%")
    with open(f"logs/losses_{tag}.txt", "a") as file:
        file.write(f"SVM Test Accuracy: {accuracy_percentage:.2f}%\n")


    # Create a confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)

    # Plot the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, cmap='Blues')

    # Add labels and title to the plot
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'images/cm_{tag}.png')
    plt.show()

