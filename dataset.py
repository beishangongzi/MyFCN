# encoding: utf-8
"""
@author: FroyoZzz
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: froyorock@gmail.com
@software: garner
@file: dataset.py
@time: 2019-08-07 17:21
@desc:
"""
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from Transforms import Squeeze


class ObtTrainDataset(Dataset):
    def __init__(self, image_path=r"data/obt/image", mode="train"):
        assert mode in ("train", "val", "test")
        self.image_path = image_path
        self.image_list = glob(os.path.join(self.image_path, "*.npy"))
        self.mode = mode

        if mode in ("train", "val"):
            self.mask_path = self.image_path

        self.transform_x = T.Compose(
            [T.ToTensor()])
        self.transform_mask = T.Compose([T.ToTensor(), Squeeze()])

    def __getitem__(self, index):
        if self.mode in ("train", "val"):
            image_name = os.path.basename(self.image_list[index])

            X = np.load(os.path.join(self.image_path, image_name))
            masks = np.load(os.path.join(self.image_path + "Masks", image_name))
            X = X / 1.0
            masks = masks / 1.0
            X = self.transform_x(X)
            masks = self.transform_mask(masks)
            X = X.type(torch.FloatTensor)
            masks = masks.type(torch.FloatTensor)
            return X, masks

        else:
            image_name = os.path.basename(self.image_list[index])
            X = np.load(os.path.join(self.image_path, image_name))
            X = X / 1.0
            X = self.transform_x(X)
            X = X.type(torch.FloatTensor)
            return X, image_name

    def __len__(self):
        return len(self.image_list)


def test():
    train_data = ObtTrainDataset()
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(train_labels[0].dtype)


if __name__ == '__main__':
    test()
