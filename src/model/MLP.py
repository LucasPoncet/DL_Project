import matplotlib.pyplot as plt
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from ClassesData.WineDataModule import DatasetLoader
from Utils.Utilities import Utilities
from ClassesML.MLP import MLP
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerTabular import TrainerClassifier


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
root = os.path.join(os.path.dirname(__file__), "WineData")
dataset_loader = DatasetLoader(root=root, target_col="label", num_cols=None,
                               onehot_cols=["region", "station", "cepage"],
                               valid_frac=0.2, dtype=torch.float32)
train_dataset, val_dataset, test_dataset = dataset_loader.load_tabular_data()
input_dim = train_dataset[0][0].shape[0] + dataset_loader.onehot_dim
n_classes = len(set(train_dataset[2].numpy()))
# Define hyperparameters


hyperparameter = dict(input_dim=input_dim,
                      output_dim=n_classes,
                      hidden_layers_size=[64,128,256],
                      activation="relu",
                      batch_normalization=False,
                      dropout_rate=0.05,
                      learning_rate=0.001,
                      max_epoch=50)

model = MLP(hyperparameter).to(device)
scope = ScopeClassifier(model, hyperparameter)


# Utilities.images_as_canvas(images=images)
# input_size = (128, hyperparameter["input_dim"][0],
#               hyperparameter["input_dim"][1],
#               hyperparameter["input_dim"][2])

# input_data = torch.rand(size=input_size, device=device)
# print(summary(model=model,input_data=input_data, depth=5))

x_train = train_dataset[0]
y_train = train_dataset[1]
x_valid = val_dataset[0]
y_valid = val_dataset[1]

trainer = TrainerClassifier(hyperparameter=hyperparameter)
trainer.set_model(model=model,device=device)
trainer.set_scope(scope=scope)
trainer.set_data(x_train=x_train,y_train=y_train,x_valid=x_valid,y_valid=y_valid)
trainer_accuracy_list,valid_accuracy_list = trainer.run()

plt.figure()
plt.plot(trainer_accuracy_list,'b',label="Train accuracy")
plt.plot(valid_accuracy_list,'r',label="Validation accuracy")
plt.title("Train and validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()


x = np.concatenate([x_valid[n]for n in range(len(x_valid))])
x=torch.from_numpy(x).to(device)
y= np.concatenate([y_valid[n]for n in range(len(y_valid))])
y=torch.from_numpy(y).to(device)

y_hat = model(x)

y_cpu = y.cpu().detach().numpy()
y_hat_cpu = y_hat.cpu().detach().numpy()
