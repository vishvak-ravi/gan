import torch
from torch import nn
from torch.utils.data import DataLoader
from models import GeneratorNet, DiscriminatorNet
from utils import get_mnist_data_loader, LATENT_DIM, get_basic_mnist
import wandb

EPOCHS = 100
G_LEARNING_RATE = 2e-4
D_LEARNING_RATE = G_LEARNING_RATE
GENERATOR_STEPS_PER_DISCRIMINATOR_STEP = 1 # remnant of my desparation before i saw the GANhacks repo...
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainGAN(
    generator: GeneratorNet,
    discriminator: DiscriminatorNet,
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

    generator.to(device)
    discriminator.to(device)
    bce.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        dataloader = train_dataset
        # basic_dataloader = get_basic_mnist(BATCH_SIZE)
        for batch, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            # noisy_ones = torch.full(
            #     (X.shape[0], 1), 0.975 - 0.075 * torch.rand(1).item()
            # ).to(device)
            # noisy_zeros = torch.full((X.shape[0], 1), 0.05 * torch.rand(1).item()).to(
            #     device
            # )
            zeros = torch.zeros(real_imgs.shape[0], 1).to(device)
            ones = torch.ones(real_imgs.shape[0], 1).to(device)

            # optimize discriminator
            discriminator_optimizer.zero_grad()  # clear gradients
            noise = torch.randn(
                real_imgs.shape[0], LATENT_DIM, device=device
            )  # sample noise
            generated = generator(noise)  # generate from noise

            d_generator = discriminator(generated.detach())  # discriminate generated
            d_data = discriminator(real_imgs)  # discriminate data

            discriminator_loss = bce(d_data, ones) + bce(
                d_generator, zeros
            )  # discriminator aims for d(generated) = 0 and d(data) = 1
            # discriminator_loss = d_loss_fn(d_generator, d_data)

            # discriminator_loss = discriminator_loss_fn(
            #    d_generator, d_data
            # )  # evaluate discriminator perf.
            discriminator_loss.backward()  # compute gradients
            discriminator_optimizer.step()  # optimize weights
            
            # # Zero out .grad variables in discriminator network (otherwise we would have corrupt results)
            # discriminator_optimizer.zero_grad()

            # # -log(D(x)) <- we minimize this by making D(x)/discriminator_net(real_images) as close to 1 as possible
            # real_discriminator_loss = bce(discriminator(real_imgs), ones)

            # # G(z) | G == generator_net and z == utils.get_gaussian_latent_batch(batch_size, device)
            # noise = torch.randn(real_imgs.shape[0], LATENT_DIM, device=device)
            # fake_images = generator(noise)
            # # D(G(z)), we call detach() so that we don't calculate gradients for the generator during backward()
            # fake_images_predictions = discriminator(fake_images.detach())
            # # -log(1 - D(G(z))) <- we minimize this by making D(G(z)) as close to 0 as possible
            # fake_discriminator_loss = bce(fake_images_predictions, zeros)

            # discriminator_loss = real_discriminator_loss + fake_discriminator_loss
            # discriminator_loss.backward()  # this will populate .grad vars in the discriminator net
            # discriminator_optimizer.step()  # perform D weights update according to optimizer's strategy
            
            # optimize generator
            for _ in range(GENERATOR_STEPS_PER_DISCRIMINATOR_STEP):
                generator_optimizer.zero_grad()
                noise = torch.randn(real_imgs.shape[0], LATENT_DIM, device=device)
                generated = generator(noise)
                d_generator = discriminator(generated)

                generator_loss = -bce(
                    d_generator, zeros
                )  # generator aims for d(generated) = 1
                generator_loss.backward()
                generator_optimizer.step()

            # wandb logging
            if batch % 100 == 0:
                if BATCH_SIZE == 1:  # original paper suggested this weirdly enough
                    sample_img = generator(noise)
                else:
                    rand_img_idx = torch.randint(BATCH_SIZE - 1, (1,))
                    sample_img = generator(noise)
                wandb.log(
                    {
                        "generator_loss": generator_loss.item(),
                        "discriminator_loss": discriminator_loss.item(),
                        "d_data": torch.mean(d_data).item(),
                        "d_generated": torch.mean(d_generator).item(),
                        "generated_sample": wandb.Image(sample_img),
                    }
                )

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
    generator = GeneratorNet()
    discriminator = DiscriminatorNet()
    mnist_loader = get_mnist_data_loader(batch_size=BATCH_SIZE)
    trainGAN(generator, discriminator, mnist_loader)
