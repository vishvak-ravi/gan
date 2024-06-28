import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from model import GeneratorZero, DiscriminatorZero, GeneratorLoss, DiscriminatorLoss
from data import MNISTTrainDataLoader
from torch import nn

TRAINING_ITERATIONS = 10
K = 1  # discriminator steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainGAN(
    untrained_generator: GeneratorZero,
    untrained_discriminator: DiscriminatorZero,
    train_dataset: DataLoader,
    epochs: int,
):
    generator_loss_fn: GeneratorLoss = GeneratorLoss()
    generator_optimizer = torch.optim.SGD(untrained_generator.parameters(), lr=1e-3, momentum=0.9)
    discriminator_loss_fn: DiscriminatorLoss = DiscriminatorLoss()
    discriminator_optimizer = torch.optim.SGD(
        untrained_discriminator.parameters(), lr=1e-3, momentum=0.9
    )
    untrained_generator.to(device)
    untrained_discriminator.to(device)
    generator_loss_fn.to(device)
    discriminator_loss_fn.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        dataloader = train_dataset
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)

            # optimize discriminator
            noise = torch.randn(X.shape[0], 64 * 64, device=device)  # sample noise
            generated = untrained_generator(noise)  # generate from noise
            d_generator = untrained_discriminator(generated)  # discriminate generated
            d_data = untrained_discriminator(X)  # discriminate data
            discriminator_loss = discriminator_loss_fn(
                d_generator, d_data
            )  # evaluate discriminator perf.
            discriminator_loss.backward()  # compute gradients
            discriminator_optimizer.step()  # optimize weights
            discriminator_optimizer.zero_grad()  # clear gradients

            # optimize generator
            noise = torch.randn(X.shape[0], 64 * 64, device=device)
            generated = untrained_generator(noise)
            d_generator = untrained_discriminator(generated)
            generator_loss = generator_loss_fn(d_generator, True)
            generator_loss.backward()
            generator_optimizer.step()
            generator_optimizer.zero_grad()
            print(f"finished batch {batch}")
            if batch % 64 == 0:
                sample_image = generated[0]
                sample_image_path = f"samples/ZERO/epoch{epoch}_batch{batch}.png"
                save_gen(sample_image, sample_image_path)
        print(f"Saving epoch: {epoch} states")
        torch.save(
            untrained_generator.state_dict(), f"model_weights/generator_{epoch}.pth"
        )
        print(f"Saved generator: {epoch} states")
        torch.save(
            untrained_discriminator.state_dict(),
            f"model_weights/discriminator_{epoch}.pth",
        )
        print(f"Saved discriminator: {epoch} states")
    print("Done!")


def save_gen(x: torch.tensor, save_path):
    x = x.squeeze().cpu()
    to_pil_img = ToPILImage()
    img = to_pil_img(x)
    img.save(save_path)


if __name__ == "__main__":
    generator = GeneratorZero()
    discriminator = DiscriminatorZero()
    mnist_loader = MNISTTrainDataLoader(batch_size=4)
    trainGAN(generator, discriminator, mnist_loader, epochs=5)
