import torch
from torch import nn
from utils.utils import MNIST_IMAGE_SIZE, LATENT_DIM

LOG_EPSILON = 1e-7
DISCRIMINATOR_LAYER_SIZES = [128, 256, 512, 1024]
GENERATOR_LAYER_SIZES = DISCRIMINATOR_LAYER_SIZES
MAXOUT_K = 5

class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.batch_size = args['batch_size']
        self.batch_norm = args['batch_norm']
        self.relu_slope = args['relu_slope']
        
        def layer(input_dim, output_dim, normalize=self.batch_norm, activation=nn.LeakyReLU(self.relu_slope)):
            layer = [nn.Linear(input_dim, output_dim)]
            if normalize:
                layer.append(nn.BatchNorm1d(output_dim))
            layer.append(activation)
            return layer
        
        self.model = nn.Sequential(
            nn.Flatten(),
            *layer(LATENT_DIM, GENERATOR_LAYER_SIZES[0], normalize=False),
            *layer(GENERATOR_LAYER_SIZES[0], GENERATOR_LAYER_SIZES[1]),
            *layer(GENERATOR_LAYER_SIZES[1], GENERATOR_LAYER_SIZES[2]),
            *layer(GENERATOR_LAYER_SIZES[2], GENERATOR_LAYER_SIZES[3]),
            *layer(GENERATOR_LAYER_SIZES[3], MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE, normalize=False, activation=nn.Tanh()),
        )

    
    def forward(self, x):
        x = self.model(x)
        return torch.reshape(x, (self.batch_size, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    
class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.batch_size = args['batch_size']
        self.batch_norm = args['batch_norm']
        self.relu_slope = args['relu_slope']
        self.dropout = nn.Dropout(args['dropout'])
        
        def layer(input_dim, output_dim, normalize=self.batch_norm, activation=nn.LeakyReLU(self.relu_slope)):
            layer = [nn.Linear(input_dim, output_dim)]
            if normalize:
                layer.append(nn.BatchNorm1d(output_dim))
            layer.append(activation)
            return layer
        
        self.model = nn.Sequential(
            nn.Flatten(),
            *layer(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE, DISCRIMINATOR_LAYER_SIZES[0], normalize=False),
            *layer(DISCRIMINATOR_LAYER_SIZES[0], DISCRIMINATOR_LAYER_SIZES[1], normalize=False),
            self.dropout,
            *layer(DISCRIMINATOR_LAYER_SIZES[1], DISCRIMINATOR_LAYER_SIZES[2], normalize=False),
            self.dropout,
            *layer(DISCRIMINATOR_LAYER_SIZES[2], 1, normalize=False, activation=nn.Sigmoid()),
        )

    def forward(self, x):
        x = self.model(x)
        return torch.reshape(x, (self.batch_size, 1))
    
## ablations
## batch_size: 1/64/256
## SGD vs Adam 
## leakyReLU vs ReLU // 0 vs 0.2
## batch_normalization // 0/1
## dropout // 0 vs 0.2 vs 0.5
## data normalization // 0/1