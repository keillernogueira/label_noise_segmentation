# Noisy Segmentation for Kaggle challenge

## Context
[Kaggle Challenge](https://www.kaggle.com/competitions/data-centric-land-cover-classification-challenge-2/overview)

## Data
Folder structure:

* **data/external**
    * **dataset**
        * `noisy_labels`
        * `image_patches`
    * **experiments**
        * `refinetnet/version_0`
        * `refinetnet/version_1`
        * ...

## Getting started
Get the original dataset (or manually download it from Kaggle and put it in):
```
mkdir -p data/external & cd data/external
# If you never used Kaggle you will need to get your API key: https://www.kaggle.com/docs/api
kaggle competitions download -c data-centric-land-cover-classification-challenge
unzip data-centric-land-cover-classification-challenge.zip
rm data-centric-land-cover-classification-challenge.zip
```

Creating conda environment:
```
conda create -y -n <your_env> & conda activate <your_env>
conda install pip
pip install -r requirements.txt
```
Training:
```
cd src
python main_train.py
```