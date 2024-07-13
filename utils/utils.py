import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader

MNIST_ROOT = "data"
MNIST_IMAGE_SIZE = 28
LATENT_DIM = 128

def get_basic_mnist(batch_size: int):
    mnist_dataset = MNIST(
            "data", train=True, download=True, transform=ToTensor()
    )
    mnist_dataloader = DataLoader(
        mnist_dataset, batch_size=batch_size
    )
    return mnist_dataloader

# from https://github.com/soumith/ganhacks normalization is *highly* beneficial
def get_mnist_data_loader(args):
    batch_size = args.batch_size
    data_norm = args.data_norm
    
    mnist_transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))]) if data_norm else None
    mnist_dataset = MNIST(
        MNIST_ROOT, train=True, transform=mnist_transform, download=True
    )
    mnist_dataloader = DataLoader(
        mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return mnist_dataloader