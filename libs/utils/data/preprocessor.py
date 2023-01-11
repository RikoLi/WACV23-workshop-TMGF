from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
import random
import math
from PIL import Image

import torch.utils.data as data
import random

class ProxySampleSet(Dataset):
    def __init__(self, args, fnames) -> None:
        super().__init__()
        self.fnames = fnames
        self.transform = T.Compose([
                T.Resize((args.height, args.width), interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(10),
                T.RandomCrop((args.height, args.width)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.5, value=[0.485, 0.456, 0.406])
            ])

    def __getitem__(self, index: int):
        fn = self.fnames[index]
        img = Image.open(fn).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.fnames)

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        #fname, pid, camid = self.dataset[index]
        input_data = self.dataset[index]
        fname = input_data[0]
        pid = input_data[1]
        camid = input_data[2]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index



class CameraAwarePreprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(CameraAwarePreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pseudo_label, camid, img_index, accum_label = self.dataset[index]

        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pseudo_label, camid, img_index, accum_label

