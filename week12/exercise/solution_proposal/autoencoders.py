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

"""Implements three variants of an autoencoder"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

def config():
    """Return a dict of configuration settings used in the program"""

    conf = {}

    # Whether you are training or testing
    conf['mode'] = 'train' # {'train', 'test'}
    # Select autoencoder variant in {'compression', 'denoising', 'sparse'}
    conf['variant'] = 'compression'

    conf['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    conf['job_dir'] = Path('/tmp/{}_autoencoder/'.format(conf['variant']))
    # Relevant dataset will be put in this location after download
    conf['data_dir'] = Path('/tmp/mnist_data')
    # Location to place checkpoints
    conf['checkpoint_dir'] = conf['job_dir'].joinpath('train/checkpoints')
    # Location to place output
    conf['output_dir'] = conf['job_dir'].joinpath('{}/output'.format(conf['mode']))
    # Path of checkpoint you want to restore variables from
    conf['restore_path'] = conf['checkpoint_dir'].joinpath('checkpoint_010000.pt')

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
    conf['loss_function'] = 'mean_squared_error'
    # The number of epochs to run before terminating training
    conf['num_epochs'] = 50
    # The batch size used in training.
    conf['batch_size'] = 256
    # The step size used by the optimization routine.
    conf['learning_rate'] = 1.0e-2

    # Variant-specific hyperparameters
    # For the denoising autoencoder
    conf['gauss_std'] = 0.2
    # For sparse autoencoder. \rho in the lecture slides
    conf['sparsity'] = 0.01
    # Regularization strength. For sparse autoencoder
    conf['sparse_loss_weight'] = 1.0

    # How often (in steps) to log the training progress (to stdout)
    conf['monitor_progress'] = 100
    # How often (in steps) to save checkpoints
    conf['periodic_checkpoint'] = 1000
    # How often (in steps) to write images during training
    conf['monitor_mosaic'] = 500

    # How many test results to show in a plotted mosaic at the end of training
    conf['mosaic_height'] = 4
    conf['mosaic_width'] = 5

    return conf


def gaussian_noise(mean, std, shape):
    """Add gaussian noise to tensor and clip the result in value range [min_val, max_val]

    returns the torch equivalent of

        np.random.normal(mean, std, in_array.shape)
    """
    mean = mean * torch.ones(shape)
    std = std * torch.ones(shape)
    return torch.distributions.normal.Normal(mean, std).sample()


def kl_divergence(rho, rho_hat):
    """
    Computes the KL divergence between two Bernoulli distributions p and q

        D_KL(p||q) = sum_x p(x) ( log p(x) / q(x) )

    where

        p ~ Bernoulli(rho)
        q ~ Bernoulli(rho_hat)

    """
    return (
        rho * np.log(rho) - rho * rho_hat.log() +
        (1.0 - rho) * np.log(1.0 - rho) - (1.0 - rho) * (1.0 - rho_hat).log()
        )


def reconstruction_loss(name, references, predictions):
    """
    Return a loss function value given the input which is expected to be two tensors of the same
    size (examples, dimension). Inputs are expected to have values in (0, 1)
    """
    if name == 'mean_squared_error':
        loss = (references - predictions).pow(2).mean()
    elif name == 'cross_entropy':
        epsilon = 1e-10 # To avoid log(0)
        loss = -(
            references * (epsilon + predictions).log() +
            (1.0 - references) * (epsilon + 1.0 - predictions).log()
            ).mean()
    return loss


def sparsity_loss(latent_vector, rho):
    """Return loss function on the sparsity in the latent layer"""
    rho_hat = latent_vector.mean(dim=0) # Average over all examples
    loss = kl_divergence(rho, rho_hat).sum()
    return loss


class Net(torch.nn.Module):
    """Definition of the autoencoder"""

    def __init__(self, conf):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(conf['height'] * conf['width'], 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 128)
        self.fc4 = torch.nn.Linear(128, conf['height'] * conf['width'])
        self.latent_vector = None
        self.prediction = None


    def encoder(self, x):
        """Encoder part of the autoencoder"""
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


    def decoder(self, x):
        """Decoder part of the autoencoder"""
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


    def forward(self, x):
        """Defines one forward pass of the autoencoder"""
        self.latent_vector = self.encoder(x)
        self.prediction = self.decoder(self.latent_vector)
        return self.prediction


def plot_mosaic(images, mosaic_height, mosaic_width, filename):
    """Write a mosaic of mosaic_height*mosaic_width images to filename"""
    plt.figure(figsize=(mosaic_width, mosaic_height))
    for ind in range(mosaic_height * mosaic_width):
        image = images[ind, :, :].cpu().detach().numpy()
        plt.subplot(mosaic_height, mosaic_width, ind+1)
        plt.imshow(image, origin='upper', cmap='gray', clim=(0.0, 1.0))
        plt.axis('off')
    plt.savefig(filename)


def train(conf, model):
    """Training the autoencoder"""
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
            if conf['variant'] == 'denoising':
                noise = gaussian_noise(0.0, conf['gauss_std'], input_batch.shape).to(conf['device'])
                noisy_input_batch = torch.clamp(input_batch + noise, 0.0, 1.0)
                predictions = model.forward(noisy_input_batch)
            else:
                predictions = model.forward(input_batch)
            loss = reconstruction_loss(conf['loss_function'], input_batch, predictions)
            if conf['variant'] == 'sparse':
                loss += (
                    conf['sparse_loss_weight'] *
                    sparsity_loss(model.latent_vector, conf['sparsity'])
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
                input_batch = input_batch.view(-1, conf['height'], conf['width'])
                predictions = predictions.view(-1, conf['height'], conf['width'])
                if conf['variant'] == 'denoising':
                    input_batch = noisy_input_batch.view(-1, conf['height'], conf['width'])
                plot_mosaic(
                    input_batch,
                    conf['mosaic_height'],
                    conf['mosaic_width'],
                    conf['output_dir'].joinpath(iter_str + '_original.png'),
                    )
                plot_mosaic(
                    predictions,
                    conf['mosaic_height'],
                    conf['mosaic_width'],
                    conf['output_dir'].joinpath(iter_str + '_reconstructed.png'),
                    )
            if total_iter % conf['periodic_checkpoint'] == 0:
                ckpt_path = conf['checkpoint_dir'].joinpath('checkpoint_' + iter_str + '.pt')
                print("Writing checkpoint to {}".format(ckpt_path))
                torch.save(model.state_dict(), ckpt_path)


def test(conf, model):
    """Running inference on a trained autoencoder"""
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            conf['data_dir'],
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            ),
        batch_size=conf['batch_size'],
        shuffle=True,
        )
    model.load_state_dict(torch.load(conf['restore_path']))
    model.eval()
    running_loss = 0
    num_images = 0
    num_iter = 0
    for total_iter, (input_batch, _) in enumerate(data_loader, 1):
        input_batch = input_batch.to(conf['device'])
        iter_str = '{:>06}'.format(total_iter)
        if conf['variant'] == 'denoising':
            noise = gaussian_noise(0.0, conf['gauss_std'], input_batch.shape).to(conf['device'])
            input_batch = torch.clamp(input_batch + noise, 0.0, 1.0)
        input_batch = input_batch.view(-1, conf['height'] * conf['width'])
        predictions = model.forward(input_batch)
        loss = reconstruction_loss(conf['loss_function'], input_batch, predictions)
        if conf['variant'] == 'sparse':
            loss += (
                conf['sparse_loss_weight'] *
                sparsity_loss(model.latent_vector, conf['sparsity'])
                )
        running_loss += loss
        if total_iter % conf['monitor_mosaic'] == 0:
            print(
                "Eval {:>6} Avg loss {:>6.4f}"
                .format(total_iter * conf['batch_size'], running_loss / total_iter)
                )
            input_batch = input_batch.view(-1, conf['height'], conf['width'])
            predictions = predictions.view(-1, conf['height'], conf['width'])
            plot_mosaic(
                input_batch,
                conf['mosaic_height'],
                conf['mosaic_width'],
                conf['output_dir'].joinpath(iter_str + '_original.png'),
                )
            plot_mosaic(
                predictions,
                conf['mosaic_height'],
                conf['mosaic_width'],
                conf['output_dir'].joinpath(iter_str + '_reconstructed.png'),
                )
        num_images += input_batch.shape[0]
        num_iter = total_iter
    print("Finished inference on {} images".format(num_images))
    print("Resulting average loss = {:>6.4f}".format(running_loss / num_iter))


def main():
    """Main"""
    print("Start program")
    conf = config()
    model = Net(conf).to(conf['device'])
    if conf['mode'] == 'train':
        train(conf, model)
    else:
        test(conf, model)

if __name__ == '__main__':
    main()
