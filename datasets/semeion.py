import os
import sys 
sys.path.append(os.path.dirname(__file__)) 

import numpy as np
import torch
from tqdm import tqdm
from random import shuffle
from PIL import Image


class Semeion(torch.utils.data.Dataset):
    """Semeion dataset"""

    def __init__(self, root="semeion/", train = True, transform=None, download = False,  seed=42):
        """
        Args:
            train (bool): If true returns training set, else test
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            seed (int): seed used for train/test split
        """
        self.seed = seed
        self.size = [16, 16]
        self.num_channels = 1
        self.num_classes = 10
        self.root_dir = root
        self.transform = transform
        self.train = train
        self._load_data()

    def _load_data(self):
        """Loads the data from the passed root directory. Splits in test/train based on seed. By default resized to 256,256
        """
        fname = os.path.join(self.root_dir, "semeion/semeion.data")
        file = open(fname, 'r')
        lines = file.readlines()
        size = self.size[0] * self.size[1]

        images = [];
        labels = [];
        fnumber = 0;

        for line in lines:
            data = line.split(' ')
            image = [];
            label = [];

            for i in range(0, size):
                image.append(int(float(data[i])))
            images.append(image)

            for i in range(size, size + self.num_classes):
                label.append(int(float(data[i]))) 
            labels.append(label)

            fnumber += 1

        #Shuffle data
        images_shuffle = []
        labels_shuffle = []
        indexes = list(range(len(images)))
        shuffle(indexes)
        for i in indexes:
            images_shuffle.append(images[i])
            labels_shuffle.append(labels[i])

        images = images_shuffle
        labels = labels_shuffle

        samples = len(lines)

        train_samples = 1300
        test_samples = 1100


        if self.train:
            self.data = np.array(images[:train_samples], dtype=np.uint8)*255
            self.data = self.data.reshape(train_samples, self.size[0], self.size[1])
            self.targets = np.argmax(np.array(labels[:train_samples], dtype=np.uint8), axis = 1)
        else:
            self.data = np.array(images[test_samples:], dtype=np.uint8)*255
            self.data = self.data.reshape(samples - test_samples, self.size[0], self.size[1])
            self.targets = np.argmax(np.array(labels[test_samples:], dtype=np.uint8), axis = 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, self.targets[idx]