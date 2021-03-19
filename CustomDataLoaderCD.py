from __future__ import print_function, division
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import torch
import random
import cv2


def clip_and_normalize(ct_slice, min_clip_value, max_clip_value):
    """
    This method clips ct slice between minimum and maximum values and normalize it between 0 and 1
    :param min_clip_value:
    :param ct_slice: numpy img array
    :return: clipped and normalized img array
    """
    ct_slice_clip_norm = (ct_slice - min_clip_value) / (max_clip_value - min_clip_value)
    ct_slice_clip_norm[ct_slice_clip_norm < 0] = 0
    ct_slice_clip_norm[ct_slice_clip_norm > 1] = 1

    return ct_slice_clip_norm


def norm(x):
    return (x.astype(float) - 128) / 128


class Bms(Dataset):
    """
    Class represent the dataset
    """

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dt = {'id': 'str', 'label': 'int'}
        self.annotation = pd.read_csv(csv_file, sep=',', dtype=dt)

        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        """

        :param idx:
        :return: an element of dataset
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.annotation.iloc[idx, 0]
        image = cv2.imread(path)
        y = int(self.annotation.iloc[idx, 1])
        sample = {'image': image, 'label': y}
        if self.transform:
            sample = self.transform(sample)

        return sample


def check_even(x):
    """
    :param x: int number
    :return:
    """
    if x % 2 == 1:
        x1 = int(((x - 1) / 2) + 1)
        x2 = int((x - 1) / 2)
        return x1, x2
    else:
        x1 = int(x / 2)
        x2 = int(x / 2)
        return x1, x2


class Aug(object):

    def __call__(self, sample):
        img = sample['image']
        prob = random.random()
        if prob > 0.5:
            new_img = np.fliplr(img)
        else:
            new_img = img

        return {'image': new_img, 'label': sample['label']}


class NormPad(object):

    def __call__(self, sample):
        """

        :param sample: dictionary with the image and the label
        :return: the dictionary with the image normalized and padded
        """
        img = sample['image']
        img = norm(img)
        if np.shape(img) != (224, 224,3):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        return {'image': img, 'label': sample['label']}


class ToTensor(object):
    def __call__(self, sample):
        """
        :param sample: dictionary image -label
        :return: dictionary with the image converted to tensor adn the label
        """
        arr = sample['image']
        arr = arr.transpose(2, 0, 1)

        image = torch.from_numpy(arr).float()

        #image = image.unsqueeze(0)

        return {'image': image, 'label': sample['label']}
