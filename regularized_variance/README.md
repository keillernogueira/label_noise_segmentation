# Noisy Segmentation for Kaggle challenge

## Context

[Kaggle Challenge](https://www.kaggle.com/competitions/data-centric-land-cover-classification-challenge-2/overview)

## Getting started

Get the original dataset (or manually download it from Kaggle and put it in data/dataset):

```
mkdir -p data/ & cd data/
# If you never used Kaggle you will need to get your API key: https://www.kaggle.com/docs/api
kaggle competitions download -c data-centric-land-cover-classification-challenge
unzip data-centric-land-cover-classification-challenge.zip
rm data-centric-land-cover-classification-challenge.zip
```

Get SpaceNet2, and place it under `data/spacenet2`

Install the conda environment for Pangaea (see readme inside the Pangaea directory)

Generate spacenet labels by running `convert_spacenet.py`

###Training:

Tune ScaleMAE by running a Pangaea training with the right config:

```
TODO instructions
```

Finetune ScaleMAE on the challenge dataset:

```
TODO instructions
```

Train overregularized networks:

```
TODO instructions
```

Create submission scores:

```
python calculate_scores.py
```
