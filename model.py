import torch
from torch import nn

class GeneratorZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64 * 64, 128 * 128)
        self.linear2 = nn.Linear(128 * 128, 256 * 256)
        self.linear3 = nn.Linear(256 * 256, 128 * 128)
        self.linear4 = nn.Linear(128 * 128, 1 * 28 * 28)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        x = torch.reshape(x, (1, 28, 28))
        return x
    
#class MaxoutLayer(nn.)

class MaxoutLayer(nn.Module):
    def __init__(self, k: int, input_shape: int, output_shape: int):
        self.k_models = torch.tensor([nn.Linear(input_shape, output_shape) for _ in range(k)])
    def forward(self, x):
        k_activations = torch.tensor([layer(x) for layer in self.k_models])
        x = max(k_activations)

class DiscriminatorZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool1d()
        self.dropout = nn.Dropout1d()
        self.sequential = nn.Sequential(
            nn.Linear((1, 28, 28), 128 * 128),
            
        )
        
        self.linear1 = 
        self.linear2 = nn.Linear(128 * 128 , 64 * 64)
        self.linear3 = nn.Linear(64 * 64 , 16 * 16)
        self.linear4 = nn.Linear(16 * 16, 1)
    def forward(self, x):
        sequential = nn.Sequential(
            self.maxpool()
        )
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d()

    def forward(self, x):
        x = self.flatten(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear0 = nn.Linear((1, 28, 28), (1, 64, 64))
        self.linear1 = nn.Linear((1, 64, 64), (1, 128, 128))
        # self.conv1 = nn.Conv2d(1, 1, 5)
        self.linear2 = nn.Linear((1, 128, 128), (1, 64, 64))
        # self.conv2 = nn.Conv2d(1,1,5)
        self.linear3 = nn.Linear((1, 64, 64), (1, 28, 28))

    def forward(self, x):
        x = self.linear0(x)
        x = self.sigmoid(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        x = self.relu(x)
