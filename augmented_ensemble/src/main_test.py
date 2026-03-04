import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
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
root_logger = logging.getLogger('pytorch_lightning')
root_logger.setLevel(logging.INFO)
fmt = '[%(levelname)s] - %(asctime)s - %(name)s: %(message)s (%(filename)s:%(funcName)s:%(lineno)d)'
root_logger.handlers[0].setFormatter(logging.Formatter(fmt))
logger = logging.getLogger('pytorch_lightning.core')


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
        logger.info(f'No checkpoint provided; using the pretrained model.')

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
    test_the_pretrained_model = False

    if test_the_pretrained_model:
        checkpoint_dir = Path('../data/external/experiments/refinenet/version_pretrained/dummy')
        settings_fp = './configs/default_settings.yaml'
        ckpt = None
    else:
        checkpoint_dir = Path('../data/external/experiments/refinenet/version_1/checkpoints')
        settings_fp = checkpoint_dir.parent / 'settings.yaml'

        # get the best model checkpoint
        ckpt = get_best_model_ckpt(checkpoint_dir)

    with open(settings_fp, 'r') as fp:
        all_settings = yaml.load(fp, Loader=yaml.FullLoader)
        logger.info(all_settings)

        if test_the_pretrained_model:
            # disable the aleatoric uncertainty for the pretrained model
            all_settings['task']['loss'] = {'name': 'BCELoss', 'args': None}

    # test the model on each fold
    use_no_split = not all_settings['data']['use_cv_split'] and all_settings['data']['val_size_f'] == 0
    folds = ['train'] if use_no_split else ['train', 'valid', 'test']
    for fold in folds:
        # extract the epoch number from the checkpoint name
        use_tta = all_settings['task']['use_tta']
        num_samples = 1 if not use_tta else all_settings['task']['num_samples']
        output_dir = checkpoint_dir.parent / 'results' / f"ckpt_best_tta_{use_tta}_n_{num_samples}" / fold / 'preds'
        test_model(settings=all_settings, checkpoint=ckpt, fold=fold, output_dir=output_dir)
