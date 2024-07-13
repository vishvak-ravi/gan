import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from utils.utils import get_mnist_data_loader, LATENT_DIM

import argparse
import wandb


def trainGAN(
    generator,
    discriminator,
    train_dataset: DataLoader,
    args
):
    #argument parsing
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    OPTIMIZER =  args.optimizer
    DISCRIMINATOR_LR = args.d_learning_rate
    GENERATOR_LR = args.g_learning_rate
    RELU_SLOPE = args.relu_slope
    DATA_NORM = args.data_norm
    DROPOUT = args.dropout
    GENERATOR_STEPS_PER_DISCRIMINATOR_STEP = args.generator_steps_per_discriminator_step
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # wandb setup
    wandb.login()
    run = wandb.init(
        project="gan",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "discriminator_lr": DISCRIMINATOR_LR,
            "generator_lr": GENERATOR_LR,
            "relu_slope": RELU_SLOPE,
            "data_normalize":DATA_NORM,
            "dropout": DROPOUT,
        },
    )

    wandb.watch(generator, log_freq=100, idx=0)
    wandb.watch(discriminator, log_freq=100, idx=1)
    
    def sample_noise():
        return torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
    
    if OPTIMIZER == "Adam":
        generator_optimizer = torch.optim.Adam(
            generator.parameters(), lr=GENERATOR_LR, betas=(0.5, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=(0.5, 0.999)
        )
    elif OPTIMIZER == "SGD":
        generator_optimizer = torch.optim.SGD(
            generator.parameters(), lr=GENERATOR_LR, momentum=0.9
        )
        discriminator_optimizer = torch.optim.SGD(
            discriminator.parameters(), lr=DISCRIMINATOR_LR, momentum=0.9
        )
    else:
        raise ValueError("Unknown optimizer or unsupported")

    bce = nn.BCELoss()

    generator.to(DEVICE)
    discriminator.to(DEVICE)
    bce.to(DEVICE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        dataloader = train_dataset
        # basic_dataloader = get_basic_mnist(BATCH_SIZE)
        for batch, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(DEVICE)
            
            #use noisy labels for gt's
            noisy_ones = torch.full((real_imgs.shape[0], 1), 0.975 - 0.075 * torch.rand(1).item()
            ).to(DEVICE)
            zeros = torch.zeros((real_imgs.shape[0], 1)).to(DEVICE)
            ones = torch.ones((real_imgs.shape[0], 1)).to(DEVICE)

            # optimize discriminator
            discriminator_optimizer.zero_grad()  # clear gradients
            noise = sample_noise()
            generated = generator(noise)  # generate from noise

            d_generator = discriminator(generated.detach())  # discriminate generated
            d_data = discriminator(real_imgs)  # discriminate data

            discriminator_loss = bce(d_data, ones) + bce(
                d_generator, zeros
            )  # discriminator aims for d(generated) = 0 and d(data) = 1

            discriminator_loss.backward()  # compute gradients
            discriminator_optimizer.step()  # optimize weights
            
            # optimize generator
            for _ in range(GENERATOR_STEPS_PER_DISCRIMINATOR_STEP):
                generator_optimizer.zero_grad()
                noise = sample_noise()
                generated = generator(noise)
                d_generator = discriminator(generated)

                generator_loss = -bce(
                    d_generator, zeros
                )  # generator aims for d(generated) = 1
                generator_loss.backward()
                generator_optimizer.step()

            # wandb logging
            if batch % 100 == 0:
                sample_img = generator(sample_noise())
                wandb.log(
                    {
                        "generator_loss": generator_loss.item(),
                        "discriminator_loss": discriminator_loss.item(),
                        "d_data": torch.mean(d_data).item(),
                        "d_generated": torch.mean(d_generator).item(),
                        "generated_sample": wandb.Image(sample_img),
                    }
                )
    
    #save weights
    torch.save(generator.state_dict(), f"model_weights/generator_{epoch}.pth")
    print(f"Saved generator")
    torch.save(
        discriminator.state_dict(),
        f"model_weights/discriminator_{epoch}.pth",
    )
    print(f"Saved discriminator")
    print("Finished training")

def parse_args():
    parser = argparse.ArgumentParser(description="Ablations for GAN")
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='Adam')
    parser.add_argument('--relu_slope', type=float, help='LeakyReLU slope (0 for ReLU)', default=0.2)
    parser.add_argument('--batch_norm', type=int, help='Batch normalization (0 or 1)', default=1)
    parser.add_argument('--dropout', type=float, help='Dropout rate', default=0.2)
    parser.add_argument('--data_norm', type=int, help='Data normalization (0 or 1)', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--g_learning_rate', type=float, help='Learning rate for the generator', default=2e-4,)
    parser.add_argument('--d_learning_rate', type=float, help='Learning rate for the discriminator', default=2e-4,)
    parser.add_argument('--generator_steps_per_discriminator_step', type=int, help='Generator steps per discriminator step', default=1,)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generator = Generator(args)
    discriminator = Discriminator(args)
    mnist_loader = get_mnist_data_loader(args)
    trainGAN(generator, discriminator, mnist_loader, args)
