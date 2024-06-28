import torch
from torch import nn


class GeneratorZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(64 * 64, 64 * 64)
        self.linear2 = nn.Linear(64 * 64, 128 * 128)
        self.linear3 = nn.Linear(128 * 128, 64 * 64)
        self.linear4 = nn.Linear(64 * 64, 1 * 28 * 28)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        x = torch.reshape(x, (batch_size, 28, 28))
        return x


class Maxout(nn.Module):
    def __init__(self, k: int, input_shape: int, output_shape: int):
        super().__init__()
        self.k_models = nn.ModuleList(
            [nn.Linear(input_shape, output_shape) for _ in range(k)]
        )
        self.dropout = nn.Dropout1d()

    def forward(self, x):
        k_activations = [layer(x) for layer in self.k_models]
        k_activations = torch.vstack(k_activations)
        x = torch.max(k_activations, dim=0).values
        return x


class DiscriminatorZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout()
        self.sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64 * 64),
            Maxout(5, 64 * 64, 128 * 128),
            nn.Linear(128 * 128, 64 * 64),
            self.dropout,
            nn.Linear(64 * 64, 16 * 16),
            Maxout(5, 16 * 16, 16 * 16),
            self.dropout,
            nn.Linear(16 * 16, 4 * 4),
            Maxout(5, 4 * 4, 1),
        )

    def forward(self, x):
        return self.sequential(x)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_generator, d_data):
        return torch.log(d_data) + torch.log(1 - d_generator)


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_g, early_training):
        if early_training:
            return -1 * torch.log(d_g)
        else:
            return torch.log(1 - d_g)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv1d()

#     def forward(self, x):
#         x = self.flatten(x)

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.linear0 = nn.Linear((1, 28, 28), (1, 64, 64))
#         self.linear1 = nn.Linear((1, 64, 64), (1, 128, 128))
#         # self.conv1 = nn.Conv2d(1, 1, 5)
#         self.linear2 = nn.Linear((1, 128, 128), (1, 64, 64))
#         # self.conv2 = nn.Conv2d(1,1,5)
#         self.linear3 = nn.Linear((1, 64, 64), (1, 28, 28))

#     def forward(self, x):
#         x = self.linear0(x)
#         x = self.sigmoid(x)
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         x = self.sigmoid(x)
#         x = self.linear3(x)
#         x = self.relu(x)
