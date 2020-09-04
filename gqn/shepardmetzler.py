import os, gzip
import numpy as np
import torch
from torch.utils.data import Dataset


def transform_viewpoint(v):
    """
    Transforms the viewpoint vector into a consistent
    representation
    """
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


class ShepardMetzler(Dataset):
    """
    Shepart Metzler mental rotation task
    dataset. Based on the dataset provided
    in the GQN paper. Either 5-parts or
    7-parts.
    :param root_dir: location of data on disc
    :param train: whether to use train of test set
    :param transform: transform on images
    :param fraction: fraction of dataset to use
    :param target_transform: transform on viewpoints
    """
    def __init__(self, root_dir, train=True, transform=None, fraction=1.0, dataset_folder_length=900, target_transform=transform_viewpoint):
        super(ShepardMetzler, self).__init__()
        assert fraction > 0.0 and fraction <= 1.0
        prefix = "train" if train else "test"
        #print(root_dir)
        self.root_dir = os.path.join(root_dir, prefix)
        #print(self.root_dir)
        #print(os.listdir(self.root_dir))
        
        root_dir_fldr = os.listdir(self.root_dir)
        root_dir_fldr = [f for f in root_dir_fldr if ".tfrecord" not in f]
        recs = []
        #print(self.root_dir)
        #print(root_dir_fldr)
        for fldr in (root_dir_fldr[:]):
          #print(fldr)
          dir_fldr = os.path.join(self.root_dir, fldr)
          #print(dir_fldr)
          recs.append([p for p in os.listdir(dir_fldr) if "pt" in p])
        self.records = sorted([item for sublist in recs for item in sublist])
        self.records = self.records[:int(len(self.records)*fraction)]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, self.records[idx][:3], self.records[idx])
        #print(scene_path)
        with gzip.open(scene_path, "r") as f:
            data = torch.load(f)
            images, viewpoints = list(zip(*data))
        
        images = np.stack(images)[:8]
        viewpoints = np.stack(viewpoints)[:8]
        #print(images.shape, viewpoints.shape)
        # uint8 -> float32
        images = images.transpose(0, 1, 4, 2, 3)
        images = torch.FloatTensor(images)/255.

        if self.transform:
            images = self.transform(images)

        viewpoints = torch.FloatTensor(viewpoints)
        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)

        return images, viewpoints

