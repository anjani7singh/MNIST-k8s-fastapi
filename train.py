# Importing dependencies
import torch
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
from model import MNISTClassifier

mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment("MNIST Digit Classifier")

def load_data():
    """
    Loading the dataset from the torchvision datasets.
    downloads and extracts the dataset to the data folder
    """
    # Loading Data:
    # Defining transforms--> to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    #downloading and loading the train dataset:
    train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=transform)
    # loading and transforming the test dataset:
    test_dataset = datasets.MNIST(root="data", download=True, train=False, transform=transform)
    # Create train dataloader:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # Create test dataloader:
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    #return both:
    return train_loader, test_loader

def train(epochs:int):
    with mlflow.start_run():
        #get dataloader:
        train_loader, test_loader = load_data()
        # Create an instance of the image classifier model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = MNISTClassifier().to(device)
        # Define the optimizer and loss function
        optimizer = Adam(classifier.parameters(), lr=0.005)
        mlflow.log_param('lr',0.005)
        loss_fn = nn.CrossEntropyLoss()
        # Train the model
        for epoch in range(epochs):  # Train for epochs
            train_loss=0.0
            valid_loss=0.0
            classifier.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()  # Reset gradients
                outputs = classifier(images)  # Forward pass
                loss = loss_fn(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                train_loss+=loss.item() #update epoch loss
            #validation:
            classifier.eval()
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device) #move tensors to device
                outputs = classifier(images) #predict
                loss=loss_fn(outputs,labels)
                valid_loss+=loss.item()
            # Calculating loss over entire batch size for every epoch
            train_loss = train_loss/len(train_loader)
            valid_loss = valid_loss/len(test_loader)
            print(f"Epoch:{epoch} Train loss: {train_loss} Validation loss: {valid_loss}")
            mlflow.log_metrics({'train loss': train_loss, 'valid loss': valid_loss})
        #model_info = mlflow.pytorch.log_model(classifier, "mnist_model")
        torch.save(classifier.state_dict(), 'model_state.pt')

if __name__=='__main__':
    epochs=20
    train(epochs=epochs)








