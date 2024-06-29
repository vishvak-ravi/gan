import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class MNISTTrainDataLoader(DataLoader):
    def __init__(self, batch_size=64):
        train_mnist = datasets.MNIST(
            "data", train=True, download=True, transform=ToTensor()
        )
        super().__init__(train_mnist, batch_size=batch_size)
