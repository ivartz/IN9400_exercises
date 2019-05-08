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

"""Implementation of simple t-SNE"""

from pathlib import Path
import time

import torch
import torchvision
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np


def config():
    """Return a dict of configuration settings used in the program"""

    conf = {}

    # Paths for outputs
    conf['job_dir'] = Path('/tmp/tsne/')
    # Relevant dataset will be put in this location after download
    conf['data_dir'] = Path('/tmp/mnist_data')
    # Location to place output
    conf['output_dir'] = conf['job_dir'].joinpath('output')

    # Create directories
    if not conf['output_dir'].exists():
        conf['output_dir'].mkdir(parents=True)

    conf['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    conf['height'] = 28
    conf['width'] = 28

    conf['num_iter'] = 700
    conf['learning_rate'] = 0.1

    conf['sigma'] = 10

    conf['num_images'] = 1000

    # The number of dimensions that we are mapping data to. Visualization is only implemented for 2
    conf['map_dimensions'] = 2

    # How often (in steps) to log the training progress (to stdout)
    conf['monitor_progress'] = 1
    conf['write_snapshot'] = 1

    return conf


def plot_scatter(data, labels, filename):
    """
    Scatter plot of data.

    Args:
        data: numpy array of shape (n, 2) with n (x, y) coordinates
        labels: a list of length n with labels corresponding to the data
        filename: full filepath of where to write the scatter plot
    """
    _, ax = plt.subplots()
    cmap = get_cmap('tab10')
    for label in sorted(np.unique(labels)):
        indices = np.where(labels == label)
        ax.scatter(data[indices, 0], data[indices, 1], color=cmap(label), label=label)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.0)
    plt.savefig(filename, bbox_inches='tight')


def pairwise_distances(data):
    """
    data is a tensor of size m x n, that is, a tensor with m
    datapoints, each with n dimensions. This function computes the L2 distance between point i and
            j for all points i=1, ..., m.

    To understand how it does it, consider a simplified version where n = 1:

    data = [1, 2, 3, 4]

    If you take the difference between this vector and its transpose, data - data^t

                    [ 1 ]
                    [ 2 ]
                    [ 3 ]
    [1, 2, 3, 4] -  [ 4 ]

    you get (via broadcasting)

    [ 0,  1,  2,  3]
    [-1,  0,  1,  2]
    [-2, -1,  0,  1]
    [-3, -2, -1,  0]

    The same method is used below, except datapoints are n-dimensional
    """
    num_points, num_dims = data.shape
    # np.expand_dims(data, axis=0)
    data_i = data.unsqueeze(0).expand(num_points, num_points, num_dims)
    # np.expand_dims(data, axis=1)
    data_j = data.unsqueeze(1).expand(num_points, num_points, num_dims)
    dist_ij = (data_i - data_j).pow(2).sum(dim=2).squeeze()
    return dist_ij


def asymmetric_gauss_neighbour_probability(data, sigma, swap_conditionality, device):
    """
    For given input data, return

    Pr(X_j = x_j | X_i = x_i) for all datapoint pairs x_i, x_j

    as described in the lecture slides page 29 and 30.
    """
    num_points, _ = data.shape
    distances = pairwise_distances(data)
    neigh_relation = distances.mul(-1/(2*sigma)).exp()
    if swap_conditionality:
        neigh_relation = neigh_relation.transpose(0, 1)
    # Delete diagonal elements
    not_diag_mask = (1 - torch.eye(num_points, dtype=torch.uint8)).to(device)
    neigh_relation = torch.masked_select(neigh_relation, not_diag_mask)\
        .view(num_points, num_points-1)
    prob = neigh_relation / (neigh_relation.sum(dim=1).view(num_points, 1))
    return prob


def symmetric_gauss_neighbour_probability(data, sigma, device):
    """
    For given input data, return

    Pr(X_i = x_i, X_j = x_j) for all datapoint pairs x_i, x_j

    as described in the lecture slides page 42
    """
    num_points, _ = data.shape
    p_j_given_i = asymmetric_gauss_neighbour_probability(data, sigma, False, device)
    p_i_given_j = asymmetric_gauss_neighbour_probability(data, sigma, True, device)
    p_joint_ij = (p_j_given_i + p_i_given_j) / (2 * num_points)
    return p_joint_ij


def symmetric_student_t_neighbour_probability(data, device):
    """
    For given input data, return

    Pr(X_i = x_i, X_j = x_j) for all datapoint pairs x_i, x_j

    as described in the lecture slides page 44
    """
    num_points, _ = data.shape
    distances = pairwise_distances(data)
    # Delete diagonal elements
    not_diag_mask = (1 - torch.eye(num_points, dtype=torch.uint8)).to(device)
    distances = torch.masked_select(distances, not_diag_mask).view(num_points, num_points-1)
    neigh_relation = distances.add(1).pow(-1)
    # Delete diagonal elements
    p_prob = neigh_relation / (neigh_relation.sum())
    return p_prob


def main():
    """Main"""
    print("Start program")
    torch.manual_seed(12345678)
    print("Random seed:", torch.initial_seed())
    conf = config()

    # Extract a single batch and use it as our data
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            conf['data_dir'],
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            ),
        batch_size=conf['num_images'],
        shuffle=False,
        )
    images, labels = next(iter(data_loader))
    images = images.view(-1, conf['height'] * conf['width'])
    images = images.to(conf['device'])

    map_points = torch.randn(
        conf['num_images'],
        conf['map_dimensions'],
        device=conf['device'],
        dtype=torch.float,
        requires_grad=True,
        )

    optimizer = torch.optim.Adam([map_points], lr=conf['learning_rate'])

    snapshot_num = 0
    running_loss = 0
    prev_time = time.time()
    for iteration in range(1, conf['num_iter']+1):
        p_ij = symmetric_gauss_neighbour_probability(images, conf['sigma'], conf['device'])
        q_ij = symmetric_student_t_neighbour_probability(map_points, conf['device'])
        loss = (p_ij * (p_ij/q_ij).log()).sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        #with torch.no_grad():
        #    map_points -= conf['learning_rate'] * map_points.grad
        #    map_points.grad.zero_()

        running_loss += loss.item()
        if iteration % conf['monitor_progress'] == 0:
            elapsed_time = time.time() - prev_time
            prev_time = time.time()
            secs_per_step = elapsed_time / conf['monitor_progress']
            print(
                "Step: {:>6} Loss: {:>7.4f} Sec/step: {:.5f}"
                .format(iteration, running_loss / conf['monitor_progress'], secs_per_step)
                )
            running_loss = 0
        if iteration % conf['write_snapshot'] == 0:
            filename = conf['output_dir'].joinpath(
                'mapping_{:03}_iter_{:03}.png'.format(snapshot_num, iteration)
                )
            plot_scatter(
                map_points.cpu().detach().numpy(),
                labels.cpu().detach().numpy(),
                filename,
                )
            snapshot_num += 1

if __name__ == "__main__":
    main()
