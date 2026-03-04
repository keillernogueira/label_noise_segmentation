from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import lightning as pl
import sklearn.model_selection
from PIL import Image
from building_footprint_segmentation.helpers.normalizer import min_max_image_net
from torch.utils.data import Dataset, DataLoader


def get_augmentations(fill_mask=0):
    geom_transform = A.ReplayCompose([
        # D4: https://albumentations.ai/docs/examples/example_d4/
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(),
        A.Affine(rotate=(-10, 10), border_mode=cv2.BORDER_CONSTANT, p=1.0, fill_mask=fill_mask),
    ])

    color_transform = A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.CLAHE(clip_limit=2.0, p=1.0),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
    ], p=0.5)

    transform = A.Compose([
        geom_transform,
        color_transform,
    ])

    return transform


def get_reveresed_augmentations(t, fill_mask=0):
    inverse_transorms = []
    # tranformations in reverse order (important for rotation)
    for t in t['replay']['transforms'][::-1]:
        name = t['__class_fullname__']
        if not t['applied']:
            continue
        if name == 'Affine':
            angle = t['params']['rotate']
            inverse_t = A.Rotate(limit=(-angle, -angle), border_mode=cv2.BORDER_CONSTANT, p=1.0, fill_mask=fill_mask)
            inverse_transorms.append(inverse_t)
        elif name == 'HorizontalFlip':
            inverse_t = A.HorizontalFlip(p=1.0)
            inverse_transorms.append(inverse_t)
        elif name == 'RandomRotate90':
            angle = 90 * t['params']['factor']
            inverse_t = A.Rotate(limit=(-angle, -angle), p=1.0)
            inverse_transorms.append(inverse_t)
        else:
            raise ValueError(f'Unknown transformation: {name}')

    return A.Compose(inverse_transorms)


class SegDataset(Dataset):
    def __init__(self, fp_imgs, fp_labels, in_memory=False, use_augmentations=False):
        self.fp_imgs = fp_imgs
        self.fp_labels = fp_labels
        self.in_memory = in_memory

        # load all images and labels in memory on the fly, if in_memory is True
        self.imgs = {}
        self.labels = {}

        # prepare the augmentations if needed
        self.aug_transform = get_augmentations() if use_augmentations else None

    def __getitem__(self, idx):
        fp = Path(self.fp_imgs[idx])

        assert fp.exists(), f'File not found: {fp}'

        if not self.in_memory or fp not in self.imgs:
            img = np.array(Image.open(fp))
            mask = (np.array(Image.open(self.fp_labels[idx])) > 0).astype(np.float32)

            if self.in_memory:
                self.imgs[fp] = img
                self.labels[fp] = mask
        else:
            img = self.imgs[fp]
            mask = self.labels[fp]

        if self.aug_transform is not None:
            augmented = self.aug_transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        # normalize the image
        img = min_max_image_net(img)

        return {'img': img, 'mask': mask, 'fp': str(fp)}

    def __len__(self):
        return len(self.fp_imgs)


class SegDataModule(pl.LightningDataModule):
    def __init__(self,
                 folder_imgs: str,
                 folder_labels: str,
                 val_size_f: float = 0.1,
                 test_size_f: float = 0.2,
                 use_cv_split: bool = False,
                 cv_n_splits: int = None,
                 cv_iter: int = None,
                 train_batch_size: int = 16,
                 val_batch_size: int = 32,
                 test_batch_size: int = 32,
                 train_shuffle: bool = True,
                 num_workers: int = 16,
                 in_memory: bool = False,
                 pin_memory: bool = False,
                 use_augmentations: bool = False,
                 seed=42):
        super().__init__()
        self.val_size_f = val_size_f
        self.test_size_f = test_size_f
        self.use_cv_split = use_cv_split
        self.cv_n_splits = cv_n_splits
        self.cv_iter = cv_iter
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
        self.in_memory = in_memory
        self.pin_memory = pin_memory
        self.use_augmentations = use_augmentations

        if self.use_cv_split:
            assert cv_n_splits is not None, 'cv_n_splits must be set if use_cv_split'
            assert self.cv_iter is not None, 'cv_iter must be set if use_cv_split'

        # the following will be set when calling setup
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.stage = None

        # prepare the filepaths
        assert Path(folder_imgs).exists(), f'Folder not found: {folder_imgs}'
        assert Path(folder_labels).exists(), f'Folder not found: {folder_labels}'

        fp_imgs = sorted(list(Path(folder_imgs).glob('*.png')))
        fp_labels = sorted(list(Path(folder_labels).glob('*.png')))

        assert len(fp_imgs) == len(fp_labels), 'Number of images and labels must match'

        # shuffle the data
        rng = np.random.default_rng(seed)
        idxs = rng.permutation(len(fp_imgs))
        fp_imgs = [fp_imgs[i] for i in idxs]
        fp_labels = [fp_labels[i] for i in idxs]

        # split the data into train, val, and test; do it based on the source id of the images (to ensure no overlap)
        groups = ['_'.join(fp.stem.split('_')[:3]) for fp in fp_imgs]

        if self.use_cv_split:
            train_valid_idx, test_idx = list(
                sklearn.model_selection.GroupKFold(n_splits=self.cv_n_splits).split(X=fp_imgs, groups=groups)
            )[self.cv_iter]
        else:
            train_valid_idx = np.arange(len(fp_imgs))
            test_idx = []

        if self.val_size_f > 0:
            train_idx, val_idx = list(sklearn.model_selection.GroupShuffleSplit(
                n_splits=1,
                test_size=self.val_size_f,
                random_state=seed
            ).split(X=train_valid_idx, groups=[groups[i] for i in train_valid_idx]))[0]

            # map back to the original indices
            train_idx = train_valid_idx[train_idx]
            val_idx = train_valid_idx[val_idx]
        else:
            train_idx = train_valid_idx
            val_idx = train_valid_idx

        self.train_fp_imgs = [fp_imgs[i] for i in train_idx]
        self.train_fp_labels = [fp_labels[i] for i in train_idx]
        self.val_fp_imgs = [fp_imgs[i] for i in val_idx]
        self.val_fp_labels = [fp_labels[i] for i in val_idx]
        self.test_fp_imgs = [fp_imgs[i] for i in test_idx]
        self.test_fp_labels = [fp_labels[i] for i in test_idx]

    def setup(self, stage: str = None):
        self.stage = stage
        if stage == 'fit' or stage is None:
            self.train_ds = SegDataset(
                fp_imgs=self.train_fp_imgs,
                fp_labels=self.train_fp_labels,
                in_memory=self.in_memory,
                use_augmentations=self.use_augmentations and (self.stage == 'fit')
            )

            self.valid_ds = SegDataset(
                fp_imgs=self.val_fp_imgs,
                fp_labels=self.val_fp_labels,
                in_memory=self.in_memory,
                use_augmentations=False
            )
        elif stage == 'test':
            self.test_ds = SegDataset(
                fp_imgs=self.test_fp_imgs,
                fp_labels=self.test_fp_labels,
                in_memory=self.in_memory,
                use_augmentations=False
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=(self.stage == 'fit')
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False
        )
