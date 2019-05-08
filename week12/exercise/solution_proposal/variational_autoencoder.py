#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of solution to exercise set for week 12                                  #
# IN4500 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.03.31                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Implements a variational autoencoder"""

import sys
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
import torchvision

def config():
    """Return a dict of configuration settings used in the program"""

    conf = {}

    # Whether you are training or testing
    # Values in {'train', 'test_reconstruction', 'test_generation', 'test_interpolation'}
    conf['mode'] = 'test_interpolation'

    conf['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    conf['job_dir'] = Path('/tmp/variational_autoencoder/')
    # Relevant dataset will be put in this location after download
    conf['data_dir'] = Path('/tmp/mnist_data')
    # Location to place checkpoints
    conf['checkpoint_dir'] = conf['job_dir'].joinpath('train/checkpoints')
    # Location to place output
    conf['output_dir'] = conf['job_dir'].joinpath('{}/output'.format(conf['mode']))
    # Path of checkpoint you want to restore variables from
    conf['restore_path'] = conf['checkpoint_dir'].joinpath('checkpoint_020000.pt')

    # Create directories
    if not conf['data_dir'].exists():
        conf['data_dir'].mkdir(parents=True)
    if not conf['checkpoint_dir'].exists():
        conf['checkpoint_dir'].mkdir(parents=True)
    if not conf['output_dir'].exists():
        conf['output_dir'].mkdir(parents=True)

    # Number of nodes in the input (the dimensions of an input example)
    conf['height'] = 28
    conf['width'] = 28

    # Implemented: {'cross_entropy', 'mean_squared_error'}
    conf['loss_function'] = 'cross_entropy'
    conf['latent_loss_weight'] = 0.001
    # The number of epochs to run before terminating training
    conf['num_epochs'] = 50
    # The batch size used in training.
    conf['batch_size'] = 128
    # The step size used by the optimization routine.
    conf['learning_rate'] = 1.0e-4

    # How often (in steps) to log the training progress (to stdout)
    conf['monitor_progress'] = 100
    # How often (in steps) to save checkpoints
    conf['periodic_checkpoint'] = 1000
    # How often (in steps) to write reconstruction mosaics during training
    conf['monitor_mosaic'] = 500

    # How many intermediate interpolation results to generate
    conf['num_interpolations'] = 11
    # How many interpolation examples to show
    conf['num_interpolation_examples'] = 10

    # How many test results to show in a plotted mosaic during training and test
    conf['mosaic_height'] = 4
    conf['mosaic_width'] = 5

    return conf


def kl_divergence(mu, log_sigma_squared):
    """
    Computes the KL divergence between two Gaussian distributions p and q

        D_KL(p||q) = sum_x p(x) ( log p(x) / q(x) )

    where

        p ~ N(mu, sigma^2)
        q ~ N(0, 1)
    """
    # Sum over all latent nodes
    return (1.0/2.0 * (mu*mu + log_sigma_squared.exp() - log_sigma_squared - 1.0)).sum(dim=1)


def reconstruction_loss(name, references, predictions):
    """
    Return a loss function value given the input which is expected to be two tensors of the same
    size (examples, dimension). Inputs are expected to have values in (0, 1)
    """
    if name == 'mean_squared_error':
        loss = (references - predictions).pow(2).mean()
    elif name == 'cross_entropy':
        # Binary cross entropy is used to compare the predictions and the reference
        epsilon = 1e-10 # To avoid log(0)
        loss = -(
            references * (epsilon + predictions).log() +
            (1.0 - references) * (epsilon + 1.0 - predictions).log()
            ).mean()
        #loss = loss.mean()
    else:
        print("Please specify an implemented loss function")
        sys.exit(1)
    return loss


def latent_loss(mu, log_sigma_squared):
    """Return latent cost function. Regularizes the spread."""
    return kl_divergence(mu, log_sigma_squared).mean()


class Net(torch.nn.Module):
    """Definition of the variational autoencoder"""

    def __init__(self, conf):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(conf['height'] * conf['width'], 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3_1 = torch.nn.Linear(64, 10)
        self.fc3_2 = torch.nn.Linear(64, 10)
        self.fc4 = torch.nn.Linear(10, 64)
        self.fc5 = torch.nn.Linear(64, 128)
        self.fc6 = torch.nn.Linear(128, conf['height'] * conf['width'])
        self.mu = None
        self.log_sigma_squared = None
        self.latent_vector = None
        self.prediction = None
        self.device = conf['device']


    def encoder(self, x):
        """Encoder part of the variational autoencoder"""
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        self.mu = self.fc3_1(x)
        # We sample log(sigma_squared) in stead of sigma_squared to allow for negative samples
        self.log_sigma_squared = self.fc3_2(x)
        standard_normal = torch.distributions.normal.Normal(
            torch.zeros(self.mu.shape), torch.ones(self.mu.shape)
            ).sample().to(self.device)
        latent_vector = self.mu + standard_normal * self.log_sigma_squared.exp().sqrt()

        return latent_vector


    def decoder(self, x):
        """Decoder part of the variational autoencoder"""
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x


    def forward(self, x):
        """Defines one forward pass of the variational autoencoder"""
        self.latent_vector = self.encoder(x)
        self.prediction = self.decoder(self.latent_vector)
        return self.prediction


def plot_mosaic(images, image_height, image_width, mosaic_height, mosaic_width, filename):
    """Write a mosaic of mosaic_height*mosaic_width images to filename"""
    plt.figure(figsize=(mosaic_width, mosaic_height))
    images = images.view(-1, image_height, image_width)
    for ind in range(mosaic_height * mosaic_width):
        image = images[ind, :, :].cpu().detach().numpy()
        plt.subplot(mosaic_height, mosaic_width, ind+1)
        plt.imshow(image, origin='upper', cmap='gray', clim=(0.0, 1.0))
        plt.axis('off')
    plt.savefig(filename)


def plot_interpolation(images, image_height, image_width, filename):
    """Write a row of images to filename"""
    plt.figure(figsize=(len(images), 1))
    for ind, image in enumerate(images):
        plt.subplot(1, len(images), ind+1)
        image = image.view(image_height, image_width)
        image = image.cpu().detach().numpy()
        plt.imshow(image, origin='upper', cmap='gray', clim=(0.0, 1.0))
        plt.axis('off')
    plt.savefig(filename)


def train(conf, model):
    """Train part"""
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            conf['data_dir'],
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            ),
        batch_size=conf['batch_size'],
        shuffle=True,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['learning_rate'])

    running_loss = 0
    total_iter = 0
    prev_time = time.time()
    for epoch_iter in range(1, conf['num_epochs']+1):
        for input_batch, _ in data_loader:
            input_batch = input_batch.to(conf['device'])
            optimizer.zero_grad()
            input_batch = input_batch.view(-1, conf['height'] * conf['width'])
            predictions = model.forward(input_batch)

            loss = (
                reconstruction_loss(conf['loss_function'], input_batch, predictions) +
                conf['latent_loss_weight'] * latent_loss(model.mu, model.log_sigma_squared)
                )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_iter += 1
            iter_str = '{:>06}'.format(total_iter)
            # Monitor progress, and store checkpoints
            if total_iter % conf['monitor_progress'] == 0:
                elapsed_time = time.time() - prev_time
                prev_time = time.time()
                images_per_sec = conf['monitor_progress'] * conf['batch_size'] / elapsed_time
                secs_per_step = elapsed_time / conf['monitor_progress']
                print(
                    "Step: {:>6} Epoch: {:>3} Loss: {:>7.4f} Im/sec: {:>7.1f} Sec/step: {:.5f}"
                    .format(
                        total_iter,
                        epoch_iter,
                        running_loss / conf['monitor_progress'],
                        images_per_sec,
                        secs_per_step,
                        )
                    )
                running_loss = 0
            if total_iter % conf['monitor_mosaic'] == 0:
                plot_mosaic(
                    input_batch,
                    conf['height'], conf['width'],
                    conf['mosaic_height'], conf['mosaic_width'],
                    conf['output_dir'].joinpath(iter_str + '_original.png'),
                    )
                plot_mosaic(
                    predictions,
                    conf['height'], conf['width'],
                    conf['mosaic_height'], conf['mosaic_width'],
                    conf['output_dir'].joinpath(iter_str + '_reconstructed.png'),
                    )
            if total_iter % conf['periodic_checkpoint'] == 0:
                ckpt_path = conf['checkpoint_dir'].joinpath('checkpoint_' + iter_str + '.pt')
                print("Writing checkpoint to {}".format(ckpt_path))
                torch.save(model.state_dict(), ckpt_path)


def test_reconstruction(conf, model):
    """Test part"""
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            conf['data_dir'],
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            ),
        batch_size=conf['batch_size'],
        shuffle=False,
        )
    model.load_state_dict(torch.load(conf['restore_path']))
    model.eval()
    running_loss = 0
    num_images = 0
    num_iter = 0
    for total_iter, (input_batch, _) in enumerate(data_loader, 1):
        iter_str = "{:>06}".format(total_iter)
        input_batch = input_batch.to(conf['device'])
        input_batch = input_batch.view(-1, conf['height'] * conf['width'])
        predictions = model.forward(input_batch)

        running_loss += (
            reconstruction_loss(conf['loss_function'], input_batch, predictions) +
            conf['latent_loss_weight'] * latent_loss(model.mu, model.log_sigma_squared)
            )

        if total_iter % conf['monitor_mosaic'] == 0:
            print(
                "Eval {:>6} Avg loss {:>6.4f}"
                .format(total_iter * conf['batch_size'], running_loss / total_iter)
                )
            plot_mosaic(
                input_batch,
                conf['height'], conf['width'],
                conf['mosaic_height'], conf['mosaic_width'],
                conf['output_dir'].joinpath(iter_str + '_original.png')
                )
            plot_mosaic(
                predictions,
                conf['height'], conf['width'],
                conf['mosaic_height'], conf['mosaic_width'],
                conf['output_dir'].joinpath(iter_str + '_reconstructed.png')
                )
        num_images += input_batch.shape[0]
        num_iter = total_iter
    print("Finished inference on {} images".format(num_images))
    print("Resulting average loss = {:>6.4f}".format(running_loss / num_iter))


def test_generation(conf, model):
    """Test part"""
    model.load_state_dict(torch.load(conf['restore_path']))
    model.eval()

    # Sample a random tensor of shape batch_size x latent layer size
    latent_vector = torch.distributions.normal.Normal(
        torch.zeros((conf['mosaic_height'] * conf['mosaic_width'], 10)),
        torch.ones((conf['mosaic_height'] * conf['mosaic_width'], 10)),
        ).sample().to(conf['device'])
    predictions = model.decoder(latent_vector)
    plot_mosaic(
        predictions,
        conf['height'], conf['width'],
        conf['mosaic_height'], conf['mosaic_width'],
        conf['output_dir'].joinpath('generated.png'),
        )


def test_interpolation(conf, model):
    """Test part"""
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            conf['data_dir'],
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            ),
        batch_size=2,
        shuffle=False,
        )
    model.load_state_dict(torch.load(conf['restore_path']))
    model.eval()
    for total_iter, (input_batch, label_batch) in enumerate(data_loader, 1):
        input_batch = input_batch.to(conf['device'])
        input_batch = input_batch.view(-1, conf['height'] * conf['width'])

        image_1 = input_batch[0, :]
        label_1 = label_batch[0]
        mu_1 = model.encoder(image_1)
        image_1 = image_1.view(conf['height'], conf['width'])

        image_2 = input_batch[1, :]
        label_2 = label_batch[1]
        mu_2 = model.encoder(image_2)
        image_2 = image_2.view(conf['height'], conf['width'])

        interpolations = [image_1]
        for num in range(conf['num_interpolations']):
            weight = num / (conf['num_interpolations'] - 1)
            latent_vector = (1.0 - weight) * mu_1 + weight * mu_2
            prediction = model.decoder(latent_vector)
            prediction = prediction.view(conf['height'], conf['width'])
            interpolations.append(prediction)
        interpolations.append(image_2)

        filename = 'interpolation_{:02}_{}_to_{}.png'.format(total_iter, label_1, label_2)
        plot_interpolation(
            interpolations,
            conf['height'], conf['width'],
            conf['output_dir'].joinpath(filename),
            )
        if total_iter >= conf['num_interpolation_examples']:
            break

def main():
    """Main"""
    print("Start program")
    conf = config()
    model = Net(conf).to(conf['device'])
    if conf['mode'] == 'train':
        train(conf, model)
    elif conf['mode'] == 'test_reconstruction':
        test_reconstruction(conf, model)
    elif conf['mode'] == 'test_generation':
        test_generation(conf, model)
    elif conf['mode'] == 'test_interpolation':
        test_interpolation(conf, model)


if __name__ == "__main__":
    main()
