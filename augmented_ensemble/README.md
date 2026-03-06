# Noisy Segmentation for Kaggle challenge

## Context

[Kaggle Challenge](https://www.kaggle.com/competitions/data-centric-land-cover-classification-challenge-2/overview)

## Data

Folder structure:

```text
* **data/external**
    * **dataset**
        * `noisy_labels`
        * `image_patches`
    * **experiments**
        * `refinetnet/version_0`
        * `refinetnet/version_1`
        * ...
```

## Getting started

Get the original dataset (or manually download it from Kaggle and put it in):

```bash
mkdir -p data/external & cd data/external
# If you never used Kaggle you will need to get your API key: https://www.kaggle.com/docs/api
kaggle competitions download -c data-centric-land-cover-classification-challenge
unzip data-centric-land-cover-classification-challenge.zip
rm data-centric-land-cover-classification-challenge.zip
```

Creating conda environment:

```bash
conda create -y -n <your_env> & conda activate <your_env>
conda install pip
pip install -r requirements.txt
```

Training (unique seed = 1 model, list of seeds = ensemble):

```bash
cd src
python main_train.py --settings configs/default_settings.yaml
```

Logs and per-epoch predictions are written into the TensorBoard logger directory configured in `logger` (default `../data/external/experiments/`). Per-run subfolders are created as `<name>/cv_iter_<i>_seed_<s>`.

Run a minimal ensemble averaging with `src/ensemble_average.py`. It requires per-run prediction NPZ files named `saved_predictions_{epoch:03d}.npz` under each run's `predictions/` folder.
Example usage:

```bash
# average epoch 000 predictions from three runs and save masks
python src/ensemble_average.py \
    --inputs ../data/external/experiments/refinenet/.../cv_iter_0_seed_42/predictions \
             ../data/external/experiments/refinenet/.../cv_iter_0_seed_43/predictions \
             ../data/external/experiments/refinenet/.../cv_iter_0_seed_44/predictions \
    --epoch 0 \
    --out ../data/external/experiments/refinenet/ensemble_epoch_000.npz \
    --threshold 0.5 --save-masks
```
