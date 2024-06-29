import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from model import GeneratorZero, DiscriminatorZero, GeneratorLoss, DiscriminatorLoss
from data import MNISTTrainDataLoader
from torch import nn
import wandb

TRAINING_ITERATIONS = 10
K = 1  # discriminator steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainGAN(
    generator: GeneratorZero,
    discriminator: DiscriminatorZero,
    train_dataset: DataLoader,
    epochs: int,
):
    generator_loss_fn: GeneratorLoss = GeneratorLoss()
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=1e-3, momentum=0.9)
    discriminator_loss_fn: DiscriminatorLoss = DiscriminatorLoss()
    discriminator_optimizer = torch.optim.SGD(
        discriminator.parameters(), lr=1e-3, momentum=0.9
    )
    generator.to(device)
    discriminator.to(device)
    generator_loss_fn.to(device)
    discriminator_loss_fn.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        dataloader = train_dataset
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)

            # optimize discriminator
            noise = torch.randn(X.shape[0], 64 * 64, device=device)  # sample noise
            generated = generator(noise)  # generate from noise
            d_generator = discriminator(generated)  # discriminate generated
            d_data = discriminator(X)  # discriminate data
            discriminator_loss = discriminator_loss_fn(
                d_generator, d_data
            )  # evaluate discriminator perf.
            discriminator_loss.backward()  # compute gradients
            discriminator_optimizer.step()  # optimize weights
            discriminator_optimizer.zero_grad()  # clear gradients

            # optimize generator
            noise = torch.randn(X.shape[0], 64 * 64, device=device)
            generated = generator(noise)
            d_generator = discriminator(generated)
            generator_loss = generator_loss_fn(d_generator, True)
            generator_loss.backward()
            generator_optimizer.step()
            generator_optimizer.zero_grad()

            # wandb logging
            if batch % 10 == 0:
                wandb.log(
                    {
                        "generator_loss": generator_loss.item(),
                        "discriminator_loss": discriminator_loss.item(),
                    }
                )
                if batch % 100 == 0:
                    wandb.log({"generated_sample": wandb.Image(generated)})

        print(f"Saving epoch: {epoch} states")
        torch.save(generator.state_dict(), f"model_weights/generator_{epoch}.pth")
        print(f"Saved generator: {epoch} states")
        torch.save(
            discriminator.state_dict(),
            f"model_weights/discriminator_{epoch}.pth",
        )
        print(f"Saved discriminator: {epoch} states")
    print("Done!")


if __name__ == "__main__":
    generator = GeneratorZero()
    discriminator = DiscriminatorZero()
    mnist_loader = MNISTTrainDataLoader(batch_size=4)
    trainGAN(generator, discriminator, mnist_loader, epochs=5)
