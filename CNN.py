import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import StepLR
from load_moonboard import load_moonboard
import matplotlib.pyplot as plt


# pretrain resnet18

grades = [4, 7, 10]
batch_size = 1000

root = 'data'
device = 'cpu'
num_classes = len(grades)
print_interval = 100

# Load data into numpy array

(x_train, y_train), (x_test, y_test) = load_moonboard(grades=grades)
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

print(load_moonboard(grades=grades)[0])

# Create train and test datasets, data loaders, etc.

tensor_train = torch.tensor(x_train)
tensor_train = tensor_train.type(torch.float32)
train_labs = y_train
train_labs = train_labs.astype(np.int64)
tensor_train_labs = torch.tensor(train_labs)

tensor_test = torch.tensor(x_test)
tensor_test = tensor_test.type(torch.float32)
test_labs = y_test
test_labs = test_labs.astype(np.int64)
tensor_test_labs = torch.tensor(test_labs)

train_dataset = TensorDataset(tensor_train, tensor_train_labs)
test_dataset = TensorDataset(tensor_test, tensor_test_labs)

# Define the grid search parameters
params_grid = {
    'momentum': [0.9, 0.95, 0.99],
    'num_epochs': [10, 20, 30],
    'batch_size': [100, 500, 1000]
}

# Define optimizer and loss function
loss_func = nn.CrossEntropyLoss()

# Perform the grid search
best_acc = 0
for momentum in params_grid['momentum']:
    for num_epochs in params_grid['num_epochs']:
        for batch_size in params_grid['batch_size']:
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            # Specify NN
            model = nn.Sequential(
                nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(32, 32, 9, padding=4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Dropout2d(p=0.3),
                nn.Flatten(),
                nn.Linear(64, 2048), nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(2048, num_classes)
            )
            model.to(device)

            # Define optimizer and loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
            
            ### Train NN ###
            step = 0
            losses = []
            model.train()
            for epoch in range(num_epochs):
                scheduler.step()
                for data, targets in train_loader:
                    data = data.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    logits = model(data)

                    loss = loss_func(logits, targets)

                    # L2 regularization
                    l2_reg = torch.tensor(0.).to(device)
                    for param in model.parameters():
                        l2_reg += torch.norm(param)
                    loss += 1e-5 * l2_reg

                    loss.backward()
                    optimizer.step()

                    if not (step + 1) % print_interval:
                        print('[epoch: {}, step: {}, loss: {}]'.format(epoch, step, loss.item()))
                    losses.append(loss.item())
                    step += 1

            # plt.plot(losses)
            # plt.xlabel('Step')
            # plt.ylabel('Loss')
            # plt.title('Training Loss')
            # plt.show()

            model.eval()
            conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(device)
                    targets = targets.to(device)
                    logits = model(data)
                    pred_classes = torch.argmax(logits, dim=1)

                    classes = np.sort(np.unique(targets.cpu()))
                    for i in range(len(pred_classes)):
                        target = np.array(targets[i].cpu(), dtype=np.int64)
                        pred = np.array(pred_classes[i].cpu(), dtype=np.int64)

                        row = np.where(classes == target)
                        col = np.where(classes == pred)

                        conf_matrix[row, col] += 1

            acc = np.diag(conf_matrix).sum() / conf_matrix.sum()
            if acc > best_acc:
                best_acc = acc
                best_params = {
                    'momentum': momentum,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size
                }

print('Best hyperparameters:', best_params)
print(conf_matrix)
acc = np.diag(conf_matrix).sum() / conf_matrix.sum()
print('\nTest Accuracy: {} %'.format(acc * 100))