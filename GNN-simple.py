import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from load_moonboard import load_moonboard
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

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
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

# Create adjacency matrix
adj_train = np.zeros((x_train.shape[0], x_train.shape[2], x_train.shape[3], x_train.shape[2], x_train.shape[3]), dtype=np.float32)
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[2]):
        for k in range(x_train.shape[3]):
            if j > 0:
                adj_train[i, j, k, j-1, k] = 1
            if j < x_train.shape[2] - 1:
                adj_train[i, j, k, j+1, k] = 1
            if k > 0:
                adj_train[i, j, k, j, k-1] = 1
            if k < x_train.shape[3] - 1:
                adj_train[i, j, k, j, k+1] = 1

adj_test = np.zeros((x_test.shape[0], x_test.shape[2], x_test.shape[3], x_test.shape[2], x_test.shape[3]), dtype=np.float32)
for i in range(x_test.shape[0]):
    for j in range(x_test.shape[2]):
        for k in range(x_test.shape[3]):
            if j > 0:
                adj_test[i, j, k, j-1, k] = 1
            if j < x_test.shape[2] - 1:
                adj_test[i, j, k, j+1, k] = 1
            if k > 0:
                adj_test[i, j, k, j, k-1] = 1
            if k < x_test.shape[3] - 1:
                adj_test[i, j, k, j, k+1] = 1

# Create train and test graphs
train_data = []
test_data = []
for i in range(x_train.shape[0]):
    x = torch.tensor(x_train[i])
    y = torch.tensor([y_train[i]])
    adj = torch.tensor(adj_train[i])
    train_data.append(Data(x=x, y=y, edge_index=adj.nonzero().t()))

for i in range(x_test.shape[0]):
    x = torch.tensor(x_test[i])
    y = torch.tensor([y_test[i]])
    adj = torch.tensor(adj_test[i])
    test_data.append(Data(x=x, y=y, edge_index=adj.nonzero().t()))


# Create train and test dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Define ResNet18 architecture with graph convolutional layers
class ResNetGCN(nn.Module):
    def __init__(self, num_classes):
        super(ResNetGCN, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.gcn1 = GCNConv(128, 128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.gcn2 = GCNConv(256, 256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.gcn3 = GCNConv(512, 512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.conv1(x)
        x = self.gcn1(x, edge_index)
        x = self.conv2(x)
        x = self.gcn2(x, edge_index)
        x = self.conv3(x)
        x = self.gcn3(x, edge_index)

        x = torch.mean(x, dim=(2,3))
        x = self.fc(x)
        return x

# Define loss function and optimizer
model = ResNetGCN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels, edge_index = data.x, data.y.view(-1), data.edge_index
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % print_interval == (print_interval-1):
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_interval))
            running_loss = 0.0

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels, edge_index = data.x, data.y.view(-1), data.edge_index
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d %%' % (len(test_data), 100 * correct / total))
