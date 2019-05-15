#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of solution to exercise set for week 13                                  #
# IN4500 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.05.03                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
"""
Implementation of DCGAN

https://arxiv.org/pdf/1511.06434.pdf


Implementation is influenced by the DCGAN pytorch tutorial

https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

from pathlib import Path
import argparse
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def config(job_dir, data_dir):
    """Return a dict of configurations used in this program"""

    conf = {}

    # Option in {'train', 'test_generation'}
    conf['mode'] = 'train'

    # Root folder to place program output
    conf['job_dir'] = job_dir
    # Relevant dataset will be put in this location after download
    conf['data_dir'] = data_dir
    # Location to place checkpoints
    conf['checkpoint_dir'] = conf['job_dir'].joinpath('train/checkpoints')
    # Location to place output
    conf['output_dir'] = conf['job_dir'].joinpath('{}/output'.format(conf['mode']))
    # Path of checkpoint you want to restore variables from
    conf['restore_steps'] = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # What device to run on
    conf['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Sidelength of the square input image used in training (random crop or resize to fit)
    conf['target_size'] = 64
    # Size of the latent vector that is the input to the generator
    conf['latent_length'] = 100

    # The number of epochs to run before terminating training
    conf['num_epochs'] = 100
    # The batch size used in training.
    conf['batch_size'] = 128
    # The step size used by the optimization routine.
    conf['learning_rate'] = 2.0e-4

    # How often (in steps) to log the training progress (to stdout)
    conf['monitor_progress'] = 10
    # How often (in steps) to write reconstruction mosaics during training
    conf['monitor_mosaic'] = 100
    # How often (in steps) to save checkpoints
    conf['periodic_checkpoint'] = 1000

    # Create directories
    if not conf['data_dir'].exists():
        conf['data_dir'].mkdir(parents=True)
    if not conf['checkpoint_dir'].exists():
        conf['checkpoint_dir'].mkdir(parents=True)
    if not conf['output_dir'].exists():
        conf['output_dir'].mkdir(parents=True)

    return conf


def maybe_download_lfw(download_root_dir):
    """
    Possibly download and extract images from the Labeled Faces in the Wild dataset

    http://vis-www.cs.umass.edu/lfw/
    """

    download_dir = download_root_dir.joinpath('lfw')
    if download_dir.exists() and download_dir.is_dir() and list(download_dir.iterdir()):
        print("LFW dataset already downloaded")
        return download_dir

    tar_path = download_root_dir.joinpath('lfw.tgz')
    if not tar_path.exists():
        url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
        cmd = ['curl', url, '-o', str(tar_path)]
        print("Downloading LFW from {} to {}".format(url, tar_path))
        subprocess.call(cmd)

    if not download_dir.exists():
        download_dir.mkdir()

    print("Unpacking")
    cmd = ['tar', 'xzf', str(tar_path), '-C', str(download_dir)]
    subprocess.call(cmd)

    return download_dir


def initialise_weights(submodule):
    """Use to initialise the weights of an entire (custom) torch.nn.Module.

    How to use:
    my_module = SomeModule() # A class inheriting torch.nn.Module
    my_module.apply(initialise_weights)
    """
    if (
            isinstance(submodule, torch.nn.Conv2d) or
            isinstance(submodule, torch.nn.ConvTranspose2d)
        ):
        # Initialise from a random normal distribution with mean 0.0 and stdev 0.02
        torch.nn.init.normal_(submodule.weight.data, 0.0, 0.02)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        # Initialise from a random normal distribution with mean 1.0 and stdev 0.02
        torch.nn.init.normal_(submodule.weight.data, 1.0, 0.02)
        # Initialise all bias parameters to zero
        torch.nn.init.constant_(submodule.bias.data, 0.0)


class Generator(torch.nn.Module):
    """Generator network definition"""
    def __init__(self, conf):
        super(Generator, self).__init__()
        # Input shape (c, h, w): (latent_length, 1, 1)
        self.convtr_1 = torch.nn.ConvTranspose2d(
            in_channels=conf['latent_length'],
            out_channels=512,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
            )
        self.bn_1 = torch.nn.BatchNorm2d(num_features=512)
        # Input shape (c, h, w): (512, 4, 4)
        self.convtr_2 = torch.nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            )
        self.bn_2 = torch.nn.BatchNorm2d(num_features=256)
        # Input shape (c, h, w): (256, 8, 8)
        self.convtr_3 = torch.nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            )
        self.bn_3 = torch.nn.BatchNorm2d(num_features=128)
        # Input shape (c, h, w): (128, 16, 16)
        self.convtr_4 = torch.nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            )
        self.bn_4 = torch.nn.BatchNorm2d(num_features=64)
        # Input shape (c, h, w): (64, 32, 32)
        self.convtr_5 = torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            )
        # Output shape (c, h, w): (3, 64, 64). NOTE: The spatial shape should match target_size


    def forward(self, x):
        """Forward pass"""
        x = torch.nn.functional.relu(self.bn_1(self.convtr_1(x)))
        x = torch.nn.functional.relu(self.bn_2(self.convtr_2(x)))
        x = torch.nn.functional.relu(self.bn_3(self.convtr_3(x)))
        x = torch.nn.functional.relu(self.bn_4(self.convtr_4(x)))
        x = torch.tanh(self.convtr_5(x))
        return x


class Discriminator(torch.nn.Module):
    """Discriminator network definition"""
    def __init__(self, conf):
        super(Discriminator, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        # Input shape (c, h, w): (3, 64, 64)
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            )
        # Input shape (c, h, w): (64, 32, 32)
        self.conv_2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            )
        self.bn_1 = torch.nn.BatchNorm2d(128)
        # Input shape (c, h, w): (128, 16, 16)
        self.conv_3 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            )
        self.bn_2 = torch.nn.BatchNorm2d(256)
        # Input shape (c, h, w): (256, 8, 8)
        self.conv_4 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            )
        self.bn_3 = torch.nn.BatchNorm2d(512)
        # Input shape (c, h, w): (512, 4, 4)
        self.conv_5 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
            )


    def forward(self, x):
        """Forward pass"""
        x = self.leaky_relu(self.conv_1(x))
        x = self.leaky_relu(self.bn_1(self.conv_2(x)))
        x = self.leaky_relu(self.bn_2(self.conv_3(x)))
        x = self.leaky_relu(self.bn_3(self.conv_4(x)))
        x = torch.sigmoid(self.conv_5(x))
        return x


def plot_mosaic(images, filename):
    """Write a mosaic of images to filename"""
    num_cols = 8
    num_rows = min(images.shape[0] // num_cols, 8)
    plt.figure()
    images = images[:num_rows*num_cols, :, :, :] * 0.5 + 0.5 # Invert normalisation
    image_grid = np.transpose(
        torchvision.utils.make_grid(images, padding=2).cpu().detach().numpy(),
        (1, 2, 0)
        )
    plt.axis('off')
    plt.imsave(filename, image_grid)


def train(conf, generator, discriminator):
    """Train part"""

    # To be passed to the generator and monitor its progress through training
    fixed_noise = torch.randn(64, conf['latent_length'], 1, 1, device=conf['device'])

    binary_ce = torch.nn.BCELoss()

    fake_label = 0
    real_label = 1

    discriminator_optimiser = torch.optim.Adam(
        discriminator.parameters(), lr=conf['learning_rate'], betas=(0.5, 0.999),
        )
    generator_optimiser = torch.optim.Adam(
        generator.parameters(), lr=conf['learning_rate'], betas=(0.5, 0.999),
        )

    download_dir = maybe_download_lfw(conf['data_dir'])
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            str(download_dir),
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((conf['target_size'], conf['target_size'])),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            ),
        batch_size=conf['batch_size'],
        shuffle=True,
        )

    # Lists with values to plot
    generator_losses = []
    discriminator_losses = []
    discriminator_on_real = []
    discriminator_on_fake_1 = []
    discriminator_on_fake_2 = []

    print("Start training")
    prev_time = time.time()
    step_counter = 0
    for epoch in range(1, conf['num_epochs']+1):
        for image_batch, _ in data_loader:
            if step_counter == 0:
                plot_mosaic(image_batch, conf['output_dir'].joinpath('input_example.png'))

            # ================================================================
            # Update D: minimise -[ log(D(x)) + log(1 - D(G(z))) ]
            # ================================================================

            # Discriminator forward and backward on real images
            # ----------------------------------------------------------------
            discriminator.zero_grad()
            image_batch = image_batch.to(conf['device'])
            # Set batch size in case the last batch is smaller than conf['batch_size']
            batch_size = image_batch.shape[0]
            real_label_batch = torch.full((batch_size, ), real_label, device=conf['device'])
            discriminator_output_real = discriminator.forward(image_batch).view(-1)
            discriminator_loss_real = binary_ce(discriminator_output_real, real_label_batch)
            discriminator_loss_real.backward()
            discriminator_mean_values_real = discriminator_output_real.mean().item()

            # Discriminator forward and backward on generated images
            # ----------------------------------------------------------------
            noise = torch.randn(batch_size, conf['latent_length'], 1, 1, device=conf['device'])
            fake_batch = generator.forward(noise)
            fake_label_batch = torch.full((batch_size, ), fake_label, device=conf['device'])
            discriminator_output_fake = discriminator.forward(fake_batch.detach()).view(-1)
            discriminator_loss_fake = binary_ce(discriminator_output_fake, fake_label_batch)
            discriminator_loss_fake.backward()
            discriminator_mean_values_fake_1 = discriminator_output_fake.mean().item()

            discriminator_loss = discriminator_loss_real + discriminator_loss_fake

            # Update discriminator parameters
            discriminator_optimiser.step()

            # ================================================================
            # Update G: minimise -[ log(D(G(z))) ]
            # ================================================================
            generator.zero_grad()
            # Since we updated the discriminator, pass the generated signal through once more
            discriminator_output_fake = discriminator.forward(fake_batch).view(-1)
            generator_loss = binary_ce(discriminator_output_fake, real_label_batch)
            generator_loss.backward()
            discriminator_mean_values_fake_2 = discriminator_output_fake.mean().item()

            # Update generator parameters
            generator_optimiser.step()

            if step_counter % conf['monitor_progress'] == 0:
                elapsed_time = time.time() - prev_time
                prev_time = time.time()
                images_per_sec = conf['monitor_progress'] * conf['batch_size'] / elapsed_time
                secs_per_step = elapsed_time / conf['monitor_progress']
                print(
                    "Step {:>5} Epoch {:>3} Loss_g {:>6.3f} Loss_d {:>6.3f} "
                    "D(x) {:.3f} D(G(z)) {:.3f} {:.3f} Im/sec {:>7.1f} Sec/step {:.4f}"
                    .format(
                        step_counter,
                        epoch,
                        discriminator_loss.item(),
                        generator_loss.item(),
                        discriminator_mean_values_real,
                        discriminator_mean_values_fake_1,
                        discriminator_mean_values_fake_2,
                        images_per_sec,
                        secs_per_step,
                        )
                    )

            if step_counter % conf['monitor_mosaic'] == 0:
                with torch.no_grad():
                    generated = generator.forward(fixed_noise)
                    plot_mosaic(
                        generated,
                        conf['output_dir'].joinpath(
                            'generated_example_{:06}.png'.format(step_counter)
                            )
                        )
            if step_counter > 0 and step_counter % conf['periodic_checkpoint'] == 0:
                ckpt_path = conf['checkpoint_dir'].joinpath('step_{:06}.pt'.format(step_counter))
                print("Writing checkpoint to {}".format(ckpt_path))
                torch.save(generator.state_dict(), ckpt_path)
            step_counter += 1
            discriminator_losses.append(discriminator_loss.item())
            generator_losses.append(generator_loss.item())
            discriminator_on_real.append(discriminator_mean_values_real)
            discriminator_on_fake_1.append(discriminator_mean_values_fake_1)
            discriminator_on_fake_2.append(discriminator_mean_values_fake_2)

    _, ax = plt.subplots()
    ax.plot(discriminator_losses, label='Discriminator loss')
    ax.plot(generator_losses, label='Generator loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(conf['output_dir'].joinpath('losses.png'))

    _, ax = plt.subplots()
    ax.plot(discriminator_on_real, label='Real')
    ax.plot(discriminator_on_fake_1, label='Fake 1')
    ax.plot(discriminator_on_fake_2, label='Fake 2')
    ax.set_xlabel('Step')
    ax.set_ylabel('Discriminator output')
    ax.legend()
    plt.savefig(conf['output_dir'].joinpath('discriminator_output.png'))


def test_generation(conf, generator):
    """
    Generate images from a trained generator.

    Generate examples from the same random vector at different checkpoints.
    """
    num_mosaics = 10
    noise_vectors = [
        torch.randn(64, conf['latent_length'], 1, 1, device=conf['device'])
        for _ in range(num_mosaics)
        ]
    for step in conf['restore_steps']:
        restore_path = conf['checkpoint_dir'].joinpath('step_{:06}.pt'.format(step))
        if not restore_path.exists():
            print("Warning: could not find restore path", restore_path)
            continue
        print("Restoring checkpoint from", restore_path)
        generator.load_state_dict(torch.load(restore_path))
        generator.eval()
        for ind in range(num_mosaics):
            noise = noise_vectors[ind]
            generated_images = generator.forward(noise)
            plot_mosaic(
                generated_images,
                conf['output_dir'].joinpath('generated_{:06}-{:02}.png'.format(step, ind))
                )


def main():
    """Main"""
    print("Start program")
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_dir', required=True, help='Folder to place program output')
    parser.add_argument('-d', '--data_dir', required=True, help='Folder to place downloaded data')
    args = parser.parse_args()

    # Set manual seed for reproducibility
    torch.manual_seed(12345678)

    conf = config(Path(args.job_dir), Path(args.data_dir))
    generator = Generator(conf).to(conf['device'])
    generator.apply(initialise_weights)
    discriminator = Discriminator(conf).to(conf['device'])
    discriminator.apply(initialise_weights)
    if conf['mode'] == 'train':
        train(conf, generator, discriminator)
    elif conf['mode'] == 'test_generation':
        test_generation(conf, generator)
    else:
        print("Exit without doing anything")
        return
    print("Program finished, elapsed time {:>7.1f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
