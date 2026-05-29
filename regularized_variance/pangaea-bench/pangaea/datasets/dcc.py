import pathlib
import random

import rasterio
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from pangaea.datasets.utils import DownloadProgressBar
from pangaea.datasets.base import RawGeoFMDataset


class DCC(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        subset: int,
    ):
        """Initialize the MADOS dataset.
        Link: https://marine-pollution.github.io/index.html

        Args:
            split (str): split of the dataset (train, val, test).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image.
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
        """
        super().__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        self.root_path = pathlib.Path(root_path)
        self.classes = classes
        self.split = split
        self.subset = subset % 4

        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download

        self.image_list = []
        self.target_list = []

        self.tiles = list(sorted(self.root_path.glob("training_patches/*")))

        # if split == "val":
        #     self.subset = (subset + 1) % 4
        if split == "train":
            gen = random.Random(11)
            gen.shuffle(self.tiles)
            self.tiles = self.tiles[self.subset * 1200 : self.subset * 1200 + 1200]

        self.image_list = self.tiles
        self.target_list = [
            t.parent.parent / "training_noisy_labels" / t.name for t in self.tiles
        ]

        print(f"Created DCC dataset with {len(self.target_list)} tiles.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])[
            ..., ::-1
        ].copy()  # read image, convert to RGB
        image = image.transpose(2, 0, 1)  # CHW

        image = torch.from_numpy(image.astype(np.float32))
        # images must be of shape (C T H W)
        image = image.unsqueeze(1)

        target = cv2.imread(self.target_list[index], 0)
        target = torch.from_numpy(target.astype(np.int64))

        output = {
            "image": {
                "optical": image,
            },
            "target": target,
            "metadata": {"id": str(self.tiles[index].stem)},
        }

        return output
