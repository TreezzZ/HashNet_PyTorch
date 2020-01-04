import torch
import os
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, batch_size, num_workers):
    """
    Loading nus-wide dataset.

    Args:
        root(str): Path of image files.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
    """
    query_dataloader = DataLoader(
        NusWideDataset(
            root,
            'test_img.txt',
            'test_label_onehot.txt',
            transform=query_transform(),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_dataloader = DataLoader(
        NusWideDataset(
            root,
            'train_img.txt',
            'train_label_onehot_tc21.txt',
            transform=train_transform(),
        ),
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    retrieval_dataloader = DataLoader(
        NusWideDataset(
            root,
            'database_img.txt',
            'database_label_onehot.txt',
            transform=query_transform(),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, query_dataloader, retrieval_dataloader


class NusWideDataset(Dataset):
    """
    Nus-wide dataset, 21 classes.

    Args
        root(str): Path of image files.
        img_txt(str): Path of txt file containing image file name.
        label_txt(str): Path of txt file containing image label.
        transform(callable, optional): Transform images.
    """
    def __init__(self, root, img_txt, label_txt, transform=None):
        self.root = root
        self.transform = transform

        img_txt_path = os.path.join(root, img_txt)
        label_txt_path = os.path.join(root, label_txt)

        # Read files
        with open(img_txt_path, 'r') as f:
            self.data = np.array([i.strip() for i in f])
        self.targets = np.loadtxt(label_txt_path, dtype=np.float32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index], index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

