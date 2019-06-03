import numpy as np
import sys
import torch.utils.data as data
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class datasetCIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        datapath (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.

    """
    def __init__(self, dataPath, train=True):
        self.dataPath = dataPath
        self.train    = train      # training set or test set
        self.train_list = [
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',
            'data_batch_4',
            'data_batch_5',
        ]
        self.test_list = ['test_batch']

        self.data   = []
        self.labels = []

        if self.train==True:
            for path in self.train_list:
                filePath = self.dataPath + path
                fo = open(filePath, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                fo.close()
                self.data.append(entry['data'])
                self.labels += entry['labels']
            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((50000, 3, 32, 32))
            #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            filePath = self.dataPath + self.test_list[0]
            fo = open(filePath, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            fo.close()
            self.data   = entry['data']
            self.labels = entry['labels']
            self.data = self.data.reshape((10000, 3, 32, 32))
            #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        return

    def __len__(self):
        """
        :return: The total number of samples
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        img = img.astype(np.float32)
        img = (img-128)/128
        return img, target


########################################################################################################################
class datasetFashionMNIST(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        datapath (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.

    """
    def __init__(self, dataPath, train=True):
        self.dataPath = dataPath
        self.train    = train      # training set or test set

        self.images, self.labels = load_mnist(self.dataPath, self.train)

        #cast
        self.labels = self.labels.astype(dtype=np.int64)
        self.images = self.images.astype(dtype=np.float32)

        # Normalize the data: subtract the mean image
        self.mean_image = np.mean(self.images, axis=0)
        return

    def __len__(self):
        """
        :return: The total number of samples
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.images[index], self.labels[index]
        img = (img-self.mean_image)/self.mean_image
        return img, target

def load_mnist(path, train=True):
    import os
    import gzip
    import numpy as np

    if train:
        kind = 'train'
    else:
        kind = 't10k'

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


########################################################################################################################
