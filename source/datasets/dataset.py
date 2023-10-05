"""
Load Classical image datasets.
Ising, MNIST, FashionMNIST, EMNIST, EuroSAT, Semeion dataset implemented. 
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from sklearn.utils import shuffle

import jax.numpy as jnp
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from typing import Tuple, List, Union

import pickle
from sklearn.model_selection import train_test_split

from eurosat import EuroSAT, random_split
from semeion import Semeion


def load_ising_data(root, L):
    path = os.path.join(root, "L" + str(L))
    data_path = os.path.join(path, "Ising2D_L" + str(L) + "_T=All.pkl")
    label_path = os.path.join(path, "Ising2D_L" + str(L) + "_T=All_labels.pkl")

    data = pickle.load(open(data_path, "rb"))
    labels = pickle.load(open(label_path, "rb"))

    return data, labels


def prepare_ising_data(
    data: np.array,
    labels: np.array,
    dtype: np.dtype = np.float32,
    test_size: float = 0.2,
    validation_size: int = 5000,
) -> Tuple[np.ndarray, ...]:
    """Function to prepare 2D Ising model data in such way that it is trainable with neural
    newtork. Code originally taken from `Notebook 12: Identifying Phases in the 2D Ising
    Model with TensorFlow <http://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB12_CIX-DNN_ising_TFlow.html>`__

    Args:
        data (np.array): _description_
        labels (np.array): _description_
        dtype (np.dtype, optional): _description_. Defaults to np.float32.
        test_size (float, optional): _description_. Defaults to 0.2.
        validation_size (int, optional): _description_. Defaults to 5000.

    Raises:
        ValueError: Raise error incase the case that the validation set size is
        larger than the training set size.

    Returns:
        Tuple[np.ndarray, ...]: Tuple of training and test data/labels.
    """
    # divide data into ordered, critical and disordered
    X_ordered = data[:70000, :]
    Y_ordered = labels[:70000]

    X_critical = data[70000:100000, :]
    Y_critical = labels[70000:100000]

    X_disordered = data[100000:, :]
    Y_disordered = labels[100000:]

    # define training and test data sets
    X = np.concatenate((X_ordered, X_disordered)).astype(
        dtype
    )  # np.concatenate((X_ordered,X_critical,X_disordered))
    Y = np.concatenate(
        (Y_ordered, Y_disordered)
    )  # np.concatenate((Y_ordered,Y_critical,Y_disordered))

    # pick random data points from ordered and disordered states to create the training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, train_size=1.0 - test_size
    )

    if not 0 <= validation_size <= len(X_train):
        raise ValueError(
            "Validation size should be between 0 and {}. Received: {}.".format(
                len(X_train), validation_size
            )
        )

    X_validation = X_train[:validation_size]
    Y_validation = Y_train[:validation_size]
    X_train = X_train[validation_size:]
    Y_train = Y_train[validation_size:]

    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)

    return X_train, Y_train, X_test, Y_test


def preprocess_jnp(
    classes: jnp.ndarray, trainloader: DataLoader, testloader: DataLoader
) -> Tuple[jnp.ndarray, ...]:
    """
    Load Data from PyTorch DataLoader into JAX NumPy Arrays

    Args:
        classes (jnp.ndarray) : List of integers representing data classes to be loaded.
                                If None, return all classes.
        trainloader (torch.utils.data.DataLoader) : Trainset loader.
        testloader (torch.utils.data.DataLoader) : Testset loader.

    Returns:
        Tuple[jnp.ndarray, ...]: Tuple of training and test data/labels. The outputs are
        ordered as follows:

        * ``X_train``: Training samples of shape ``(num_train, img_size, img_size,
          num_channel)``.
        * ``Y_train``: Training labels of shape ``(num_train, )``.
        * ``X_test``: Test samples of shape ``(num_test, img_size, img_size,
          num_channel)``.
        * ``Y_train``: Test labels of shape ``(num_test, )``.
    """
    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    print(classes)
    # Load data as a np.ndarray
    for i, data in enumerate(trainloader, 0):
        image, label = data
        X_train.extend(list(jnp.transpose(image.detach().numpy(), (0, 2, 3, 1))))
        Y_train.extend(list(label.detach().numpy()))

    for i, data in enumerate(testloader, 0):
        image, label = data
        X_test.extend(list(jnp.transpose(image.detach().numpy(), (0, 2, 3, 1))))
        Y_test.extend(list(label.detach().numpy()))

    X_train = jnp.array(X_train)
    Y_train = jnp.array(Y_train)

    X_test = jnp.array(X_test)
    Y_test = jnp.array(Y_test)

    if classes is not None:
        train_mask = np.isin(Y_train, classes)
        X_train = X_train[train_mask]
        Y_train = Y_train[train_mask]

        Y_train = Y_train.at[jnp.argwhere(Y_train == classes[0])].set(0.0)
        Y_train = Y_train.at[jnp.argwhere(Y_train == classes[1])].set(1.0)

        test_mask = np.isin(Y_test, classes)
        X_test = X_test[test_mask]
        Y_test = Y_test[test_mask]
        Y_test = Y_test.at[jnp.argwhere(Y_test == classes[0])].set(0.0)
        Y_test = Y_test.at[jnp.argwhere(Y_test == classes[1])].set(1.0)

    return X_train, Y_train, X_test, Y_test


def get_data(
    data: str,
    load_dir: str,
    img_size: int,
    classes: Union[List[int], jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, ...]:
    r"""Function to load training data.

    Args:
      data (str): Name of the dataset.
      load_dir (str): Root directory where the datasets are located.
      imgs_size (int): Image size to be loaded.
      classes (Union[List[int], jnp.ndarray]): List of integers representing data classes
        to be loaded. If None, return all classes. Default to None.

    Returns:
        Tuple[jnp.ndarray, ...]: Tuple of training and test labels. The outputs are
        ordered as follows:

        * ``X_train``: Training samples of shape ``(num_train, img_size, img_size,
          num_channel)``.
        * ``Y_train``: Training labels of shape ``(num_train, )``.
        * ``X_test``: Test samples of shape ``(num_test, img_size, img_size,
          num_channel)``.
        * ``Y_train``: Test labels of shape ``(num_test, )``.
    """

    if classes is not None and type(classes) is not jnp.ndarray:
        classes = jnp.array(classes)

    if data == "Ising":
        data, labels = load_ising_data(
            "/data/suchang/sy_phd/v4_QML_EO/v7_EQNNs/Ising", img_size
        )
        data = np.expand_dims(data.astype(np.float32), 3)

        X_train, Y_train, X_test, Y_test = prepare_ising_data(data, labels)
    else:
        switcher = {
            "MNIST": torchvision.datasets.MNIST,
            "FashionMNIST": torchvision.datasets.FashionMNIST,
            "EMNIST": torchvision.datasets.EMNIST,
            "EuroSAT": EuroSAT,
            "semeion": Semeion,
        }

        ds = switcher.get(data, lambda: None)
        if ds is None:
            raise TypeError("Specified data type does not exist!")

        if data == "EuroSAT":
            load_dir = "/afs/cern.ch/work/s/suchang/shared/Data/eurosat/"

            transform = transforms.Compose(
                [
                    transforms.Resize([img_size, img_size]),
                    transforms.ToTensor(),
                    transforms.Grayscale(),
                ]
            )
            eurosat = ds(root=load_dir, transform=transform)
            train_ds, test_ds = random_split(eurosat)
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize([img_size, img_size]),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                ]
            )

            if data == "EMNIST":
                train_ds = ds(
                    root=load_dir,
                    split="letters",
                    train=True,
                    download=True,
                    transform=transform,
                )
                test_ds = ds(
                    root=load_dir,
                    split="letters",
                    train=False,
                    download=True,
                    transform=transform,
                )
            else:
                train_ds = ds(
                    root=load_dir, train=True, download=True, transform=transform
                )
                test_ds = ds(
                    root=load_dir, train=False, download=True, transform=transform
                )

        trainloader = torch.utils.data.DataLoader(
            train_ds, batch_size=1024, shuffle=True
        )

        testloader = torch.utils.data.DataLoader(test_ds, batch_size=1024, shuffle=True)

        X_train, Y_train, X_test, Y_test = preprocess_jnp(
            classes, trainloader, testloader
        )

    return X_train, Y_train, X_test, Y_test
