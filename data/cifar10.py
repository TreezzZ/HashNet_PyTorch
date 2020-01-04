import os
import torch
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from data.transform import train_transform, query_transform, Onehot, encode_onehot
from PIL import Image

def load_data(root, batch_size, num_workers):
    """
    Load cifar-10 dataset.

    Args
        root(str): Path of dataset.
        batch_size(int): Batch size.
        num_workers(int): Number of data loading workers.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    root = os.path.join(root, 'images')
    train_dataloader = DataLoader(
        ImagenetDataset(
            os.path.join(root, 'train'),
            transform=train_transform(),
            target_transform=Onehot(10),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    query_dataloader = DataLoader(
        ImagenetDataset(
            os.path.join(root, 'query'),
            transform=query_transform(),
            target_transform=Onehot(10),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    retrieval_dataloader = DataLoader(
        ImagenetDataset(
            os.path.join(root, 'database'),
            transform=query_transform(),
            target_transform=Onehot(10),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, query_dataloader, retrieval_dataloader,


class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        # Assume file alphabet order is the class order
        if ImagenetDataset.class_to_idx is None:
            ImagenetDataset.classes, ImagenetDataset.class_to_idx = self._find_classes(root)

        for i, cl in enumerate(ImagenetDataset.classes):
            cur_class = os.path.join(self.root, cl)
            files = os.listdir(cur_class)
            files = [os.path.join(cur_class, i) for i in files]
            self.data.extend(files)
            self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(len(files))])
        self.targets = np.asarray(self.targets)
        self.onehot_targets = torch.from_numpy(encode_onehot(self.targets, 10)).float()
    
    def get_onehot_targets(self):
        return self.onehot_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, item

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

