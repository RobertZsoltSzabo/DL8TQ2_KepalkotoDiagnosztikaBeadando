import argparse
import traceback
import sys
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data
from torchvision.transforms import Compose, CenterCrop, Resize, Normalize, ToTensor
import numpy as np
import os
from pathlib import Path

from models import Generator, Discriminator
from datasets import GrayscaleImageFolder


ROOT = Path(__file__).parent.resolve()
IMAGE_SIZE = 128
IMAGE_CHANNELS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-vector-length", type=int, default=100)
    parser.add_argument("--generator-features", type=int, default=64)
    parser.add_argument("--discriminator-features", type=float, default=64)
    parser.add_argument("--filename-filter", type=str, default='')
    parser.add_argument("--image-extension", type=str, default='png')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta", type=float, default=0.7)
    try:
        return parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)


def train(args):
    transforms = Compose([
        Resize(IMAGE_SIZE),
        CenterCrop(IMAGE_SIZE),
        ToTensor(),
        Normalize((0.5,),(0.5,))       
    ])

    dataset = GrayscaleImageFolder(root=args.data_folder,
                                   filename_filter=args.filename_filter,
                                   image_extension=args.image_extension,
                                   transform=transforms)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    generator = Generator(in_channels=args.latent_vector_length,
                          feature_channels=args.generator_features,
                          out_channels=IMAGE_CHANNELS).to(DEVICE)

    discriminator = Discriminator(in_channels=IMAGE_CHANNELS,
                                  feature_channels=args.discriminator_features,
                                  input_size=IMAGE_SIZE).to(DEVICE)

    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta, 0.999))

    if not os.path.exists(f'{ROOT}/runs/{args.experiment_name}'):
        os.makedirs(f'{ROOT}/runs/{args.experiment_name}')

    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data.to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.latent_vector_length, 1, 1, device=DEVICE)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, args.epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            
            torch.jit.save(torch.jit.script(generator), f'{ROOT}/runs/{args.experiment_name}/generator_last.pt')
            torch.jit.save(torch.jit.script(discriminator), f'{ROOT}/runs/{args.experiment_name}/discriminator_last.pt')
            if errG == np.min(G_losses):
                torch.jit.save(torch.jit.script(generator), f'{ROOT}/runs/{args.experiment_name}/generator_best.pt')
                torch.jit.save(torch.jit.script(discriminator), f'{ROOT}/runs/{args.experiment_name}/discriminator_best.pt')


    np.save(f'{ROOT}/runs/{args.experiment_name}/generator_losses.npy', np.array(G_losses))
    np.save(f'{ROOT}/runs/{args.experiment_name}/discriminator_losses.npy', np.array(D_losses))



if __name__ == "__main__":
    args = parse_args()
    train(args)
