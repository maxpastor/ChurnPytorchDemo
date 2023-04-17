# Step 1: Defining the imports (dependencies)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import logging
import sys
import os
from sklearn.metrics import roc_auc_score
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


#Step 2: Defining the model

class Model(nn.Module):
    def __init__(self, num_features, num_neurons=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, num_neurons),  #Number of neurons of the layer before and after
            nn.LeakyReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, 1),
            nn.Sigmoid()
        )
#Step 3: Define the forward function
    def forward(self, input_data):
        return self.network(input_data)


#Step 4: Define the Train function
def train(x_input, y_labels, model, optimizer, loss_fn, device, epochs=1000):
    model.to(device)
    model.train()
    x_input = x_input.to(device)
    y_labels = y_labels.to(device)
    
    train_loss_values = []
    for epoch in range(epochs):
        prediction = model(x_input) #model.forward(x_input) works as well
        loss = loss_fn(prediction, torch.unsqueeze(y_labels, 1)) #Erorr to be minimized
        loss.backward() #Backpropagation preparation
        optimizer.step() #Update the weights
        train_loss_values.append(loss.item()) #Save the loss value. item() is used to get the value of the tensor [2.7] -> 2.7 

        if epoch % 100 == 0:
            logger.info("Train Epoch: {} \tLoss: {:.6f}".format(epoch, loss.item()))
#Step 5: Define the Validation function

def validate(x, y, model, device, loss_fn):
    with torch.inference_mode():
        model.to(device)
        x = x.to(device)
        y = y.to(device)
        model.eval()
        predictions = model(x)
        loss = loss_fn(predictions, torch.unsqueeze(y, 1)) #[[0.2], [0.3], [0.4]] -> [0.2, 0.3, 0.4]
        #We want to get the AUC score
        auc = roc_auc_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy())
        logger.info("Validation set: Average loss: {:.4f}, AUC: {:.4f}".format(loss, auc))
        return loss, auc
#Step 6: Put everything together
def main(args):
    #Grab the data (we assume that the data is ready to be consumed)
    train_df = pd.read_csv(args.data_dir + "/train.csv")
    validation_df = pd.read_csv(args.test_dir + "/validation.csv")

    num_features = train_df.shape[1] - 1 

    x_train = train_df[:, 1:] # In front of the : is where we start to grab, after is where we stop to grab 
    y_train = train_df[:, 0]

    x_validation = validation_df[:, 1:]
    y_validation = validation_df[:, 0]
    #Define some Hyperparameters
    num_neurons = args.neurons
    learning_rate = args.lr

    #define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Define the model, optimizer and loss function
    model = Model(num_features=num_features, num_neurons=num_neurons)
    loss_fn = nn.BCELoss() #Binary classification loss
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate) #Adaptive momentum (AdaM)

    #Run training 
    loss, auc = train(x_train, y_train, model, optimizer, loss_fn, device, epochs=args.epochs)

    #Run validation
    validate(x_validation, y_validation, model, device, loss_fn)

    #That's it ! 

#Step 7: Accept arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=10, metavar="N", help="Number of neurons in the hidden layers")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, metavar="E", help="Number of epochs")
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], metavar="D", help="Directory where the data is stored")
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"], metavar="T", help="Directory where the validation data is stored")
    main(parser.parse_args())