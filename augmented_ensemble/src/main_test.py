import logging
import os
from pathlib import Path
import torch
import argparse

import numpy as np
import lightning as pl
import yaml
from torch.utils.data import DataLoader

# local imports
from utils.data import SegDataModule
from utils.model import build_model
from utils.task import SegTask

# https://github.com/PyTorchLightning/pytorch-lightning/issues/5225
if 'SLURM_NTASKS' in os.environ:
    del os.environ['SLURM_NTASKS']
if 'SLURM_JOB_NAME' in os.environ:
    del os.environ['SLURM_JOB_NAME']

# Logger (console and TensorBoard)
root_logger = logging.getLogger('lightning')
root_logger.setLevel(logging.INFO)
fmt = '[%(levelname)s] - %(asctime)s - %(name)s: %(message)s (%(filename)s:%(funcName)s:%(lineno)d)'
root_logger.handlers[0].setFormatter(logging.Formatter(fmt))
logger = logging.getLogger('lightning.core')


def test_model(settings, fold, output_dir, checkpoint=None):
    # fix the seed
    seed = settings['task']['seed']
    pl.seed_everything(seed, workers=True)
    logger.info(f'Initial settings: {settings}')

    # Data
    dm = SegDataModule(**settings['data'])

    # Model
    model = build_model(settings)

    # Task
    task_params = settings['task']
    task = SegTask(model=model, task_params=task_params, outdir=output_dir)

    if checkpoint is not None:
        logger.info(f'Loading model from {checkpoint}')
        device = f"cuda:{settings['trainer']['devices'][0]}" if settings['trainer']['accelerator'] == 'gpu' else 'cpu'
        SegTask.load_from_checkpoint(
            checkpoint_path=checkpoint,
            map_location=device,
            model=model,
            task_params=task_params
        )
    else:
        logger.info('No checkpoint provided; using the pretrained model.')

    # Trainer
    trainer_dict = settings['trainer']
    trainer = pl.Trainer(**trainer_dict)

    assert fold in ('train', 'valid', 'test')
    dm.setup('test' if fold == 'test' else None)
    dl = {'train': dm.train_dataloader, 'valid': dm.val_dataloader, 'test': dm.test_dataloader}[fold]()

    test_batch_size = settings['data']['test_batch_size']
    if task_params['use_tta']:
        logger.info(f'Using TTA (n = batch_size = {dl.batch_size})')
        # duplicate the filepaths batch_size times; TODO: implement a cleaner way
        dl.dataset.fp_imgs = [fp for fp in dl.dataset.fp_imgs for _ in range(test_batch_size)]
        dl.dataset.fp_labels = [fp for fp in dl.dataset.fp_labels for _ in range(test_batch_size)]

    # get the dataset from the current dataloader and build a new dataloader
    dl = DataLoader(
        dataset=dl.dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=settings['data']['num_workers'],
    )

    # results = trainer.validate(model=task, dataloaders=dl)
    # logger.info(f'Results for {fold}: {results}')

    # export the predictions
    trainer.test(model=task, dataloaders=dl)

    results = trainer.predict(model=task, dataloaders=[dl])
    results = torch.cat(results, 0)

    labels = dl.dataset.fp_labels

    save_path = str(checkpoint).replace(".ckpt", "_predictions")

    from torchvision.utils import save_image  # Or use PIL if needed

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    for l, r in zip(labels, results):
        # Extract filename from the original label path
        filename = os.path.basename(l)
        filename = os.path.splitext(filename)[0] + "_pred.png"  # or .jpg or any format

        # Create full save path
        output_file = os.path.join(save_path, filename)

        # Save the result (assuming it's a tensor)
        if isinstance(r, torch.Tensor):
            # If r is batched, unbatch it
            if r.dim() == 4:
                r = r[0]
            save_image(r, output_file)
        else:
            print(f"Unsupported result type: {type(r)}")

def get_best_model_ckpt(checkpoint_dir, metric_name='val_BinaryF1Score', sort_method='max'):
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.exists(), f'{checkpoint_dir} not found'
    assert sort_method in ('max', 'min')

    ckpt_list = sorted(list(checkpoint_dir.glob('*.ckpt')))
    scores = np.array([float(p.stem.split(f'{metric_name}=')[1]) for p in ckpt_list if metric_name in str(p)])

    # get the index of the last best value
    sort_method_f = np.argmax if sort_method == 'max' else np.argmin
    i_best = len(scores) - sort_method_f(scores[::-1]) - 1
    ckpt_best = ckpt_list[i_best]

    return ckpt_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained models with settings file')
    parser.add_argument('--settings', type=str, default='configs/default_settings.yaml', help='Path to YAML settings file')
    parser.add_argument('--fold', type=str, default='train', choices=['train', 'valid', 'test'], help='Which split to run predictions on')
    parser.add_argument('--version', type=int, default=0, help='Model version number (default: 0)')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Optional explicit checkpoint directory')
    args = parser.parse_args()

    with open(args.settings, 'r') as fp:
        all_settings = yaml.load(fp, Loader=yaml.FullLoader)

    seed_list = all_settings['task']['seed'] if isinstance(all_settings['task']['seed'], list) else [all_settings['task']['seed']]
    cv_n = all_settings['data'].get('cv_n_splits', 1)

    for seed in seed_list:
        for cv_iter in range(cv_n):
            if args.checkpoint_dir:
                checkpoint_dir = Path(args.checkpoint_dir)
            else:
                base = Path(all_settings['logger']['save_dir'])
                name = all_settings['logger']['name']
                checkpoint_dir = base / name / f"cv_iter_{cv_iter}_seed_{seed}" / f"version_{args.version}" / "checkpoints"

            settings_fp = checkpoint_dir.parent / 'settings.yaml'
            if not checkpoint_dir.exists():
                logger.warning(f'Checkpoint dir not found: {checkpoint_dir} — skipping')
                continue

            ckpt = get_best_model_ckpt(checkpoint_dir, metric_name="train_BinaryF1Score")

            with open(settings_fp, 'r') as fp:
                run_settings = yaml.load(fp, Loader=yaml.FullLoader)

            use_tta = run_settings['task'].get('use_tta', False)
            num_samples = 1 if not use_tta else run_settings['task'].get('num_samples', 1)
            output_dir = checkpoint_dir.parent / 'results' / f"ckpt_best_tta_{use_tta}_n_{num_samples}" / args.fold / 'preds'
            test_model(settings=run_settings, checkpoint=ckpt, fold=args.fold, output_dir=output_dir)
