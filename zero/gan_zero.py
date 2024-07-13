import torch
from torch import nn
from utils.utils import MNIST_IMAGE_SIZE, LATENT_DIM

LOG_EPSILON = 1e-7
DISCRIMINATOR_LAYER_SIZES = [128, 256, 512, 1024]
GENERATOR_LAYER_SIZES = DISCRIMINATOR_LAYER_SIZES
MAXOUT_K = 5

class GeneratorZero(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.sequential(x)
        x = torch.reshape(x, (1, 28, 28))
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
            nn.Linear(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE, 256),
            nn.LeakyReLU(0.2),
            Maxout(256, 256, k=MAXOUT_K),
            self.dropout,
            nn.Linear(256, 32),
            Maxout(32, 32, k=MAXOUT_K),
            nn.LeakyReLU(0.2),
            self.dropout,
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.sequential(x)
    