import os
import time 
import pathlib
import urllib.request
import urllib.error
import zipfile

from glob import glob
import rasterio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from pangaea.datasets.utils import DownloadProgressBar
from pangaea.datasets.base import RawGeoFMDataset


class Spacenet2(RawGeoFMDataset):
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



        self.tiles = self.get_train_val_test_split(sorted(self.root_path.glob('labels/*')))

        for tile in self.tiles[self.split]:
            tile_name = str(pathlib.Path(tile).stem)
            tile_number = tile_name.split('_')[-1]
            tile_aoi = '_'.join(tile_name.split("_")[:-1])
            image_path = self.root_path / f"{tile_aoi}/PS-RGB/SN2_buildings_train_{tile_aoi}_PS-RGB_{tile_number}.tif"
            if image_path.exists():
                self.target_list.append(tile)
                self.image_list.append(image_path)

        print(f"Created Spacenet 2 dataset with {len(self.image_list)} tiles.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        image_path = self.image_list[index]
        with rasterio.open(image_path, mode='r') as src:
            image = src.read()
        image = torch.from_numpy(image.astype(np.float32))

        # images must be of shape (C T H W)
        image = image.unsqueeze(1)

        with rasterio.open(self.target_list[index], mode='r') as src:
            target = src.read(1)
        target = torch.from_numpy(target.astype(np.int64))

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {"id": str(self.tiles[self.split][index].stem)}
        }

        return output

    @staticmethod
    def get_train_val_test_split(all_files):
        # Fixed stratified sample to split data into train/val.
        # This keeps 90% of datapoints belonging to an individual event in the training set and puts the remaining 10% in the validation set.
        train, temp = train_test_split(
            all_files,
            test_size=0.2,
            random_state=23,
        )
        val, test = train_test_split(
            temp,
            test_size=0.5,
            random_state=2424,
        )

        return {"train": train, "val": val, "test": test}
