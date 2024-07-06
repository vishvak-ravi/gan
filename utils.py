import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader

MNIST_ROOT = "data"
MNIST_IMAGE_SIZE = 28
LATENT_DIM = 128


# from https://github.com/soumith/ganhacks normalization is *highly* beneficial
def get_mnist_data_loader(batch_size: int):
    mnist_transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    mnist_dataset = MNIST(
        MNIST_ROOT, train=True, transform=mnist_transform, download=True
    )
    mnist_dataloader = DataLoader(
        mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return mnist_dataloader

def get_basic_mnist(batch_size: int):
    mnist_dataset = MNIST(
            "data", train=True, download=True, transform=ToTensor()
    )
    mnist_dataloader = DataLoader(
        mnist_dataset, batch_size=batch_size
    )
    return mnist_dataloader

## not mine
def get_gan_data_transform():
    # It's good to normalize the images to [-1, 1] range https://github.com/soumith/ganhacks
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms]
    )
    return transform


def get_mnist_dataset():
    # This will download the MNIST the first time it is called
    return MNIST(
        root=MNIST_ROOT,
        train=True,
        download=True,
        transform=get_gan_data_transform(),
    )


def get_mnist_data_loader_two(batch_size):
    mnist_dataset = get_mnist_dataset()
    mnist_data_loader = DataLoader(
        mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return mnist_data_loader