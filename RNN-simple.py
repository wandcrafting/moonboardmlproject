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
x_train = np.transpose(x_train, (0, 2, 1)) # transpose the shape to (1000, 11, 18)
x_test = np.transpose(x_test, (0, 2, 1))
seq_len = x_train.shape[1]
input_dim = x_train.shape[2]

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
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device) # initialize hidden state
        out, _ = self.rnn(x, h0) # run RNN
        out = self.fc(out[:, -1, :]) # extract last hidden state and pass through linear layer
        return out

model = RNN(input_dim, 64, num_classes)
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
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

# evaluate model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs = inputs.view(-1, inputs.shape[1], inputs.shape[2]) # reshape input tensor for RNN
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print('Test Accuracy of the model on the {} test samples: {} %'.format(len(test_loader.dataset), 100 * correct / total))

