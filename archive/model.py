import torch
from torch import nn

LOG_EPSILON = 1e-7


class GeneratorZero(nn.Module):
    def __init__(self, latent_dim: int, batch_size: int):
        super().__init__()

        def layerBlock(in_feat: int, out_feat: int, normalize: bool = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize and batch_size > 1:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.sequential = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),  # TODO: 28 is a magic number WHOOPS
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = self.sequential(x)
        x = torch.reshape(x, (batch_size, 28, 28))
        return x


class Maxout(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, k: int):
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
            nn.Linear(28 * 28, 128),
            nn.LeakyReLU(0.2),
            # Maxout(512, 512, k=5),
            self.dropout,
            nn.Linear(128, 32),
            # Maxout(128, 128, k=5),
            nn.LeakyReLU(0.2),
            self.dropout,
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.sequential(x)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_generated, d_data):
        d_generated = nn.Sigmoid(d_generated)
        d_data = nn.Sigmoid(d_data)
        return torch.sum(-torch.log(d_data) / 2.0 - torch.log(1 - d_generated) / 2.0)


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, d_generated):
        d_generated = self.sigmoid(d_generated)
        return -torch.sum(torch.log(d_generated)) / 2.0


class GeneratorLossBCELogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, d_gen):
        valids = torch.full_like(d_gen, 1.0)
        return self.bce(d_gen, valids)


class DiscriminatorLossBCELogitsLabelSmoothing(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, d_generated, d_data):
        valids = torch.full_like(d_data, 0.9)
        fakes = torch.full_like(d_generated, 0.0)
        return torch.sum(self.bce(d_data, valids) + self.bce(d_generated, fakes))


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
