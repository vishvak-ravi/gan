import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from utils.utils import get_mnist_data_loader, LATENT_DIM
import wandb

EPOCHS = 100
G_LEARNING_RATE = 2e-4
D_LEARNING_RATE = G_LEARNING_RATE
GENERATOR_STEPS_PER_DISCRIMINATOR_STEP = 1 # remnant of my desparation before i saw the GANhacks repo...
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_noise():
    return torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)


def trainGAN(
    generator,
    discriminator,
    train_dataset: DataLoader,
    epochs: int = EPOCHS,
):
    # wandb setup
    wandb.login()
    run = wandb.init(
        project="gan",
        config={
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": "adam",
            "discriminator_lr": D_LEARNING_RATE,
            "generator_lr": G_LEARNING_RATE,
            "activations": "LeakyReLU",
            "data_normalize": "true",
            "dropout": 0,
        },
    )

    wandb.watch(generator, log_freq=100, idx=0)
    wandb.watch(discriminator, log_freq=100, idx=1)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=G_LEARNING_RATE, betas=(0.5, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=D_LEARNING_RATE, betas=(0.5, 0.999)
    )

    bce = nn.BCELoss()

    generator.to(DEVICE)
    discriminator.to(DEVICE)
    bce.to(DEVICE)

    for epoch in range(epochs):
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


if __name__ == "__main__":
    generator = Generator(LATENT_DIM)
    discriminator = Discriminator(batch_size=BATCH_SIZE)
    mnist_loader = get_mnist_data_loader(batch_size=BATCH_SIZE)
    trainGAN(generator, discriminator, mnist_loader)
