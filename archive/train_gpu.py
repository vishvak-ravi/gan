import torch
from torch.utils.data import DataLoader
from model import (
    GeneratorZero,
    DiscriminatorZero,
    DiscriminatorLoss,
    GeneratorLoss,
    DiscriminatorLossBCELogitsLabelSmoothing,
    GeneratorLossBCELogits,
)
from data import MNISTTrainDataLoader
from torch import nn
import wandb

EPOCHS = 200
LATENT_DIM = 128
G_LEARNING_RATE = 2e-4
D_LEARNING_RATE = 1e-5
BATCH_SIZE = 1
GENERATOR_STEPS_PER_DISCRIMINATOR_STEP = 1
D_LOSS_VERSION = 1
G_LOSS_VERSION = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainGAN(
    generator: GeneratorZero,
    discriminator: DiscriminatorZero,
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
            "discriminator_lr": D_LEARNING_RATE,
            "generator_lr": G_LEARNING_RATE,
            "d_loss_v": D_LOSS_VERSION,
            "g_loss_v": G_LOSS_VERSION,
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
    d_loss_fns = {0: DiscriminatorLoss(), 1: DiscriminatorLossBCELogitsLabelSmoothing()}
    g_loss_fns = {0: GeneratorLoss(), 1: GeneratorLossBCELogits()}

    d_loss_fn = d_loss_fns[D_LOSS_VERSION]
    g_loss_fn = g_loss_fns[G_LOSS_VERSION]

    bce = nn.BCELoss()

    generator.to(device)
    discriminator.to(device)
    bce.to(device)
    d_loss_fn.to(device)
    g_loss_fn.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        dataloader = train_dataset
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            # noisy_ones = torch.full(
            #     (X.shape[0], 1), 0.975 - 0.075 * torch.rand(1).item()
            # ).to(device)
            # noisy_zeros = torch.full((X.shape[0], 1), 0.05 * torch.rand(1).item()).to(
            #     device
            # )
            # zeros = torch.zeros(X.shape[0], 1).to(device)

            # optimize discriminator
            discriminator_optimizer.zero_grad()  # clear gradients
            noise = torch.randn(X.shape[0], LATENT_DIM, device=device)  # sample noise
            generated = generator(noise)  # generate from noise

            d_generator = discriminator(generated.detach())  # discriminate generated
            d_data = discriminator(X)  # discriminate data

            # discriminator_loss = bce(d_data, noisy_ones) + bce(
            #     d_generator, noisy_zeros
            # )  # discriminator aims for d(generated) = 0 and d(data) = 1
            discriminator_loss = d_loss_fn(d_generator, d_data)

            # discriminator_loss = discriminator_loss_fn(
            #    d_generator, d_data
            # )  # evaluate discriminator perf.
            discriminator_loss.backward()  # compute gradients
            discriminator_optimizer.step()  # optimize weights

            # optimize generator
            for _ in range(GENERATOR_STEPS_PER_DISCRIMINATOR_STEP):
                generator_optimizer.zero_grad()
                noise = torch.randn(X.shape[0], LATENT_DIM, device=device)
                generated = generator(noise)
                d_generator = discriminator(generated.detach())

                # generator_loss = -bce(
                #     d_generator, zeros
                # )  # generator aims for d(generated) = 1
                generator_loss = g_loss_fn(d_generator)

                # generator_loss = generator_loss_fn(d_generator, True)
                generator_loss.backward()
                generator_optimizer.step()

            # wandb logging
            if batch % 100 == 0:
                wandb.log(
                    {
                        "generator_loss": generator_loss.item(),
                        "discriminator_loss": discriminator_loss.item(),
                        "d_data": torch.mean(d_data).item(),
                        "d_generated": torch.mean(d_generator).item(),
                    }
                )
            if batch % 100 == 0:

                if BATCH_SIZE == 1:
                    sample_img = generated
                else:
                    rand_img_idx = torch.randint(BATCH_SIZE - 1, (1,))
                    sample_img = generated[rand_img_idx]
                wandb.log({"generated_sample": wandb.Image(sample_img)})

        if epoch % 10:
            print(f"Saving epoch: {epoch} states")
            torch.save(generator.state_dict(), f"model_weights/generator_{epoch}.pth")
            print(f"Saved generator: {epoch} states")
            torch.save(
                discriminator.state_dict(),
                f"model_weights/discriminator_{epoch}.pth",
            )
            print(f"Saved discriminator: {epoch} states")
    print("Finished training")


if __name__ == "__main__":
    generator = GeneratorZero(LATENT_DIM, BATCH_SIZE)
    discriminator = DiscriminatorZero()
    mnist_loader = MNISTTrainDataLoader(batch_size=BATCH_SIZE)
    trainGAN(generator, discriminator, mnist_loader)
