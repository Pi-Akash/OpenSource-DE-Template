# to view mlflow dashboard, type mlflow ui in cmd or powershell

# imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchmetrics import Accuracy
from torch import optim
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import pickle

# mlflow logging and experiment tracking configuration
mlflow.autolog()
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("<Your-experiment-name>")

# reading parquet file
df = pd.read_parquet("processedData.parquet")
df.drop(["CLIENTNUM"], axis = "columns", inplace = True)

labels = df["Attrition_Flag"].values
features = df.drop(["Attrition_Flag"], axis = "columns").values

class CustomerDataset(Dataset):
    """
    Dataset loader class which returns rows based on the index provided
    """
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feats = self.features[index]
        target = self.labels[index]
        return feats, target


def create_splits(features, labels, save_split = False, print_detail = False):
    """
    The function creates train, validation and test splits
    """
    data_dict = {}

    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size = 0.25, random_state = 2024)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size = 0.2, random_state = 2024)

    if print_detail:
        print("train features shape : ", x_train.shape, "train labels shape : ", y_train.shape)
        print("val features shape : ", x_val.shape, "val labels shape : ", y_val.shape)
        print("test features shape : ", x_test.shape, "test labels shape : ", y_test.shape)
        
    data_dict["train_features"] = x_train
    data_dict["train_labels"] = y_train
    data_dict["val_features"] = x_val
    data_dict["val_labels"] = y_val
    data_dict["test_features"] = x_test
    data_dict["test_labels"] = y_test

    if save_split:
        try:
            with open("data_split.pkl", "wb") as pickle_file:
                pickle.dump(data_dict, pickle_file)
            
            print("-- Pickle file written successfully.")
        except Exception as e:
            print(e)
    
    return data_dict
    
data = create_splits(features, labels, save_split = True)

x_train = torch.tensor(data["train_features"], dtype = torch.float32)
y_train = torch.tensor(data["train_labels"], dtype = torch.float32).reshape(-1, 1)

x_val = torch.tensor(data["val_features"], dtype = torch.float32) 
y_val = torch.tensor(data["val_labels"], dtype = torch.float32).reshape(-1, 1)

# creating training dataloader
train_dataset = CustomerDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

# creating validation dataloader
val_dataset = CustomerDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size = 32)

# first model with 2 Linear Layers
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(15, 6),
            nn.ReLU(),
            nn.BatchNorm1d(6),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Second Model with 3 linear Layers
class ThreeLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(15, 6),
            nn.ReLU(),
            nn.BatchNorm1d(6),
            nn.Linear(6, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    

def train(train_loader, model, loss_fn, metric_fn, optimizer, epoch):
    model.train()
    for batch, (feats, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        preds = model(feats)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        # metric on current batch
        accuracy = metric_fn(preds, targets)
        
        mlflow.log_metric("loss", loss.item(), step = epoch)
        mlflow.log_metric("accuracy", accuracy, step = epoch)
        print(f"Batch : {batch}, acc : {accuracy} [{batch} / {len(train_loader)}]")


def evaluate(val_loader, model, loss_fn, metric_fn):
    num_batches = len(val_loader)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    
    with torch.no_grad():
        for feats, targets in val_loader:
            preds = model(feats)
            eval_loss += loss_fn(preds, targets).item()
            eval_accuracy += metric_fn(preds, targets)
    
    eval_loss /= num_batches
    eval_accuracy /= num_batches
    mlflow.log_metric("eval_loss", eval_loss, step = 0)
    mlflow.log_metric("eval_accuracy", eval_accuracy, step = 0)
    
    print(f"Eval metrics : Accuracy : {eval_accuracy}, Avg. Loss: {eval_loss}")

# Experiment 1
net = TwoLayerNet()
n_epochs = 10
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.01)
acc_metric = Accuracy(task = "binary")

with mlflow.start_run() as run:
    params = {
        "epochs" : n_epochs,
        "learning_rate" : 0.01,
        "batch_size" : 32,
        "loss_function" : criterion.__class__.__name__,
        "metric_function" : acc_metric.__class__.__name__,
        "optimizer" : "Adam"
    }
    
    mlflow.log_params(params)
    
    # log model summary
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(net)))
    
    mlflow.log_artifact("model_summary.txt")
    
    for n in range(n_epochs):
        train(train_loader, net, criterion, acc_metric, optimizer, n)
    
    evaluate(val_loader, net, criterion, acc_metric)

    mlflow.pytorch.log_model(net, "model")
    
    model_scripted = torch.jit.script(net) # Export to TorchScript
    model_scripted.save('model_2_layer_10_epoch.pt') # Save


# Experiment 2
net = TwoLayerNet()
n_epochs = 20
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.01)
acc_metric = Accuracy(task = "binary")

with mlflow.start_run() as run:
    params = {
        "epochs" : n_epochs,
        "learning_rate" : 0.01,
        "batch_size" : 32,
        "loss_function" : criterion.__class__.__name__,
        "metric_function" : acc_metric.__class__.__name__,
        "optimizer" : "Adam"
    }
    
    mlflow.log_params(params)
    
    # log model summary
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(net)))
    
    mlflow.log_artifact("model_summary.txt")
    
    for n in range(n_epochs):
        train(train_loader, net, criterion, acc_metric, optimizer, n)
    
    evaluate(val_loader, net, criterion, acc_metric)

    mlflow.pytorch.log_model(net, "model")
    model_scripted = torch.jit.script(net) # Export to TorchScript
    model_scripted.save('model_2_layer_20_epoch.pt') # Save

# Experiment 3
net = ThreeLayerNet()
n_epochs = 10
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.01)
acc_metric = Accuracy(task = "binary")

with mlflow.start_run() as run:
    params = {
        "epochs" : n_epochs,
        "learning_rate" : 0.01,
        "batch_size" : 32,
        "loss_function" : criterion.__class__.__name__,
        "metric_function" : acc_metric.__class__.__name__,
        "optimizer" : "Adam"
    }
    
    mlflow.log_params(params)
    
    # log model summary
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(net)))
    
    mlflow.log_artifact("model_summary.txt")
    
    for n in range(n_epochs):
        train(train_loader, net, criterion, acc_metric, optimizer, n)
    
    evaluate(val_loader, net, criterion, acc_metric)

    mlflow.pytorch.log_model(net, "model")
    model_scripted = torch.jit.script(net) # Export to TorchScript
    model_scripted.save('model_3_layer_10_epoch.pt') # Save
    