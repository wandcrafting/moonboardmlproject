import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from load_moonboard import load_moonboard

# pretrain resnet18

grades = [4, 7, 10]
batch_size = 1000

root = 'data'
device = 'cpu'
lr = 0.01
momentum = 0.9
num_epochs = 20
print_interval = 100
num_classes = len(grades)


# Load data into numpy array

(x_train, y_train), (x_test, y_test) = load_moonboard(grades=grades)

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
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



# Specify NN
model = nn.Sequential(  # Sequential models are models where the layers are applied sequentially, in order
    nn.Flatten(),
    nn.Linear(18*11, 2048), nn.ReLU(inplace=True),
    nn.Linear(2048, 4096), nn.ReLU(inplace=True),
    nn.Linear(4096, 2048), nn.ReLU(inplace=True),
    nn.Linear(2048, num_classes))

model.to(device)  # moves model to specified device

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

### Train NN ###
step = 0  # track how many training iterations we've been through
losses = []  # will be used to store loss values at each iteration
model.train()  # sets model to training mode
for epoch in range(num_epochs):
    for data, targets in train_loader:
        # Move data and targets to the same device as the model
        data = data.to(device)
        targets = targets.to(device)

        # Zero-out gradients
        optimizer.zero_grad()

        # Run forward pass
        logits = model(data)

        # Compute loss
        loss = loss_func(logits, targets)

        # Perform backpropagation and update model parameters
        loss.backward()
        optimizer.step()

        if not (step + 1) % print_interval:
            print('[epoch: {}, step: {}, loss: {}]'.format(epoch, step, loss.item()))

        step += 1

# Compute test set predictions, confusion matrix
model.eval()
conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
with torch.no_grad():
    for data, targets in test_loader:
        # Make sure data is on the same device as the model
        data = data.to(device)
        targets = targets.to(device)

        # Run forward pass
        logits = model(data)

        # Get class predictions
        pred_classes = torch.argmax(logits, dim=1)

        classes = np.sort(np.unique(targets.cpu()))

        # Update confusion matrix
        for i in range(len(pred_classes)):
            target = np.array(targets[i].cpu(), dtype=np.int64)
            pred = np.array(pred_classes[i].cpu(), dtype=np.int64)

            row = np.where(classes == target)
            col = np.where(classes == pred)

            conf_matrix[row, col] += 1

print(conf_matrix)
acc = np.diag(conf_matrix).sum() / conf_matrix.sum()
print('\nTest Accuracy: {} %'.format(acc * 100))