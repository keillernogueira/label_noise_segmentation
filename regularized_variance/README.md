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

Install the conda environment and pangaea-bench:

```
conda env create -f environment.yml
conda activate regularized_variance
cd pangaea-bench
pip install --no-build-isolation --no-deps -e .
```

Generate SpaceNet2 labels:

```
cd ..
python3 convert_spacenet.py`
```

### Training:

Tune ScaleMAE on SpaceNet2 by running a Pangaea training with the right config:

```
cd pangaea-bench
torchrun pangaea/run.py --config-name=spacenet2 --config-path="../../configs"
```

Finetune an ensemble on the challenge dataset:

```
tuned_net=(work-dir/spacenet_pretrained/*/checkpoint__best.pth)
torchrun pangaea/run.py --config-name=challenge_dataset --config-path="../../configs" ++seed=1 ++ckpt_dir=$tuned_net ++dataset.subset=1
torchrun pangaea/run.py --config-name=challenge_dataset --config-path="../../configs" ++seed=2 ++ckpt_dir=$tuned_net ++dataset.subset=2
torchrun pangaea/run.py --config-name=challenge_dataset --config-path="../../configs" ++seed=3 ++ckpt_dir=$tuned_net ++dataset.subset=3
torchrun pangaea/run.py --config-name=challenge_dataset --config-path="../../configs" ++seed=4 ++ckpt_dir=$tuned_net ++dataset.subset=4
torchrun pangaea/run.py --config-name=challenge_dataset --config-path="../../configs" ++seed=5 ++ckpt_dir=$tuned_net ++dataset.subset=1
torchrun pangaea/run.py --config-name=challenge_dataset --config-path="../../configs" ++seed=6 ++ckpt_dir=$tuned_net ++dataset.subset=2
torchrun pangaea/run.py --config-name=challenge_dataset --config-path="../../configs" ++seed=7 ++ckpt_dir=$tuned_net ++dataset.subset=3
torchrun pangaea/run.py --config-name=challenge_dataset --config-path="../../configs" ++seed=8 ++ckpt_dir=$tuned_net ++dataset.subset=4
```

Train overregularized networks:

```
finetuned_list=(pangaea-bench/work-dir/finetuned/*/checkpoint__best.pth)
torchrun pangaea/run.py --config-name=overreg --config-path="../../configs" ++seed=1 ++ckpt_dir=${finetuned_list[0]} ++dataset.subset=1
torchrun pangaea/run.py --config-name=overreg --config-path="../../configs" ++seed=2 ++ckpt_dir=${finetuned_list[1]} ++dataset.subset=2
torchrun pangaea/run.py --config-name=overreg --config-path="../../configs" ++seed=3 ++ckpt_dir=${finetuned_list[2]} ++dataset.subset=3
torchrun pangaea/run.py --config-name=overreg --config-path="../../configs" ++seed=4 ++ckpt_dir=${finetuned_list[3]} ++dataset.subset=4
torchrun pangaea/run.py --config-name=overreg --config-path="../../configs" ++seed=5 ++ckpt_dir=${finetuned_list[4]} ++dataset.subset=1
torchrun pangaea/run.py --config-name=overreg --config-path="../../configs" ++seed=6 ++ckpt_dir=${finetuned_list[5]} ++dataset.subset=2
torchrun pangaea/run.py --config-name=overreg --config-path="../../configs" ++seed=7 ++ckpt_dir=${finetuned_list[6]} ++dataset.subset=3
torchrun pangaea/run.py --config-name=overreg --config-path="../../configs" ++seed=8 ++ckpt_dir=${finetuned_list[7]} ++dataset.subset=4

```

Create submission scores:

```
python calculate_scores.py
```
