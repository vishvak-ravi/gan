"""The original (vanilla) GAN implementation with some modifications.

    Modifications:
        The original paper used the maxout activation and dropout for regularization.
        I'm using LeakyReLU instead and batch normalization which came after the original paper was published.

    Also note that certain architectural design decisions were inspired by the DCGAN paper.
"""

import torch
from torch import nn


from utils import LATENT_DIM, MNIST_IMAGE_SIZE


def vanilla_block(in_feat, out_feat, normalize=True, activation=None):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    # 0.2 was used in DCGAN, I experimented with other values like 0.5 didn't notice significant change
    layers.append(nn.LeakyReLU(0.2) if activation is None else activation)
    return layers


class GeneratorNet(torch.nn.Module):
    """Simple 4-layer MLP generative neural network.

    By default it works for MNIST size images (28x28).

    There are many ways you can construct generator to work on MNIST.
    Even without normalization layers it will work ok. Even with 5 layers it will work ok, etc.

    It's generally an open-research question on how to evaluate GANs i.e. quantify that "ok" statement.

    People tried to automate the task using IS (inception score, often used incorrectly), etc.
    but so far it always ends up with some form of visual inspection (human in the loop).

    """

    def __init__(self, img_shape=(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)):
        super().__init__()
        self.generated_img_shape = img_shape
        num_neurons_per_layer = [
            LATENT_DIM,
            256,
            512,
            1024,
            img_shape[0] * img_shape[1],
        ]

        self.net = nn.Sequential(
            *vanilla_block(num_neurons_per_layer[0], num_neurons_per_layer[1]),
            *vanilla_block(num_neurons_per_layer[1], num_neurons_per_layer[2]),
            *vanilla_block(num_neurons_per_layer[2], num_neurons_per_layer[3]),
            *vanilla_block(
                num_neurons_per_layer[3],
                num_neurons_per_layer[4],
                normalize=False,
                activation=nn.Tanh(),
            )
        )

    def forward(self, latent_vector_batch):
        img_batch_flattened = self.net(latent_vector_batch)
        # just un-flatten using view into (N, 1, 28, 28) shape for MNIST
        return img_batch_flattened.view(
            img_batch_flattened.shape[0], 1, *self.generated_img_shape
        )


class DiscriminatorNet(torch.nn.Module):
    """Simple 3-layer MLP discriminative neural network. It should output probability 1. for real images and 0. for fakes.

    By default it works for MNIST size images (28x28).

    Again there are many ways you can construct discriminator network that would work on MNIST.
    You could use more or less layers, etc. Using normalization as in the DCGAN paper doesn't work well though.

    """

    def __init__(self, img_shape=(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)):
        super().__init__()
        num_neurons_per_layer = [img_shape[0] * img_shape[1], 512, 256, 1]

        self.net = nn.Sequential(
            *vanilla_block(
                num_neurons_per_layer[0], num_neurons_per_layer[1], normalize=False
            ),
            *vanilla_block(
                num_neurons_per_layer[1], num_neurons_per_layer[2], normalize=False
            ),
            *vanilla_block(
                num_neurons_per_layer[2],
                num_neurons_per_layer[3],
                normalize=False,
                activation=nn.Sigmoid(),
            )
        )

    def forward(self, img_batch):
        img_batch_flattened = img_batch.view(
            img_batch.shape[0], -1
        )  # flatten from (N,1,H,W) into (N, HxW)
        return self.net(img_batch_flattened)
