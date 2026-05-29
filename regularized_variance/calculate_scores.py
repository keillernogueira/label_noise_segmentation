import numpy as np
import cv2
import pathlib
import pandas as pd
import tqdm
import multiprocessing.pool

pool = multiprocessing.pool.ThreadPool(16)


def calc_iou(a, b):
    intersection = np.logical_and(a, b)
    union = np.logical_or(a, b)
    return np.sum(intersection) / np.sum(union)


class Dataset:
    def __init__(self, root_path) -> None:
        self.root_path = root_path
        self.target_list = sorted([f for f in self.root_path.glob("*")])

        print(f"Created DCC dataset with {len(self.target_list)} tiles.")

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        target = cv2.imread(self.target_list[index], 0)
        target = target.astype(np.int16)

        return target


def get_image(dataset, idx):
    return dataset[idx] / 255


def process_image(target_ds, datasets, overreg_datasets, idx):
    target = target_ds[idx]
    images = pool.starmap(get_image, zip(datasets, [idx] * len(datasets)))
    overreg_images = pool.starmap(
        get_image, zip(overreg_datasets, [idx] * len(overreg_datasets))
    )
    overreg_images = np.stack(overreg_images)

    per_model_ious = [calc_iou(target, i) for i in images]

    best_iou = max(per_model_ious)

    oreg_var = np.mean(np.var(overreg_images, axis=0))
    alpha = 1.0  # Note: This could be optimized for
    score = best_iou - (0.5 - best_iou) * oreg_var * alpha

    return score


eval_paths = list(
    pathlib.Path("pangaea-bench/work_dir/finetuned/").glob(
        "*/outputs/checkpoint__best/dcc"
    )
)

overreg_paths = list(
    pathlib.Path("pangaea-bench/work-dir/overregularized").glob(
        "*/outputs/checkpoint__best/dcc"
    )
)

print("Predictions: ", eval_paths)
print("Overregularized predictions:", overreg_paths)

data_path = pathlib.Path("data/dataset/training_noisy_labels")

eval_datasets = [Dataset(p) for p in eval_paths]
overreg_datasets = [Dataset(p) for p in overreg_paths]
print(eval_datasets)
print(overreg_datasets)
target_dataset = Dataset(data_path)

scores = []

for i, path in enumerate(tqdm.tqdm(target_dataset.target_list)):
    name = path.name
    score = process_image(target_dataset, eval_datasets, overreg_datasets, i)
    scores.append((name, score))

df = pd.DataFrame(scores, columns=["id", "score"])
df = df.sort_values(by=["score", "id"], ascending=[False, True], ignore_index=True)

print(df)

df.to_csv("./scores.csv")
df["id"].to_csv("./submission.csv")
