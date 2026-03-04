import json
import logging
import os
from pathlib import Path

import lightning as pl
import yaml

# local imports
from utils.data import SegDataModule
from utils.model import build_model
from utils.task import SegTask

# https://github.com/PyTorchLightning/pytorch-lightning/issues/5225
if 'SLURM_NTASKS' in os.environ:
    del os.environ['SLURM_NTASKS']
if 'SLURM_JOB_NAME' in os.environ:
    del os.environ['SLURM_JOB_NAME']


def train_model(settings):
    # Logger (console and TensorBoard)
    tb_logger = pl.loggers.TensorBoardLogger(**settings['logger'])
    Path(tb_logger.log_dir).mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger('lightning')
    root_logger.setLevel(logging.INFO)
    fmt = '[%(levelname)s] - %(asctime)s - %(name)s: %(message)s (%(filename)s:%(funcName)s:%(lineno)d)'
    root_logger.handlers[0].setFormatter(logging.Formatter(fmt))
    logger = logging.getLogger('lightning.core')
    fh = logging.FileHandler(Path(tb_logger.log_dir, 'console.log'))
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

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
    task = SegTask(model=model, task_params=task_params)

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_BinaryF1Score',
        filename='ckpt-{epoch:02d}-{val_BinaryF1Score:.4f}',
        save_top_k=-1,
        save_last=True,
        mode='max',
        every_n_epochs=1
    )
    summary = pl.callbacks.ModelSummary(max_depth=-1)

    # Trainer
    trainer_dict = settings['trainer']
    trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback, summary], **trainer_dict)

    # Save the config file, after adding the slurm job id (if exists)
    path_out = Path(tb_logger.log_dir, 'settings.yaml')
    setting_dict_save = settings.copy()
    setting_dict_save['SLURM_JOBID'] = os.environ['SLURM_JOBID'] if 'SLURM_JOBID' in os.environ else None
    with open(path_out, 'w') as fp:
        yaml.dump(setting_dict_save, fp, sort_keys=False)
    logger.info(f'Settings saved to to {path_out}')
    logger.info(f'Exported settings:\n{json.dumps(settings, sort_keys=False, indent=4)}')

    trainer.fit(task, dm)
    logger.info(f'Best model {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}')


if __name__ == '__main__':
    settings_fp = './configs/default_settings.yaml'

    with open(settings_fp, 'r') as fp:
        all_settings = yaml.load(fp, Loader=yaml.FullLoader)

    # set the number of workers to 0 if the data is loaded in memory
    if all_settings['data']['in_memory']:
        print('Setting the number of workers to 0')
        all_settings['data']['num_workers'] = 0

    # train a model for each CV fold
    logger_name = all_settings['logger']['name']
    if all_settings['data']['use_cv_split']:
        for cv_iter in range(all_settings['data']['cv_n_splits']):
            all_settings['data']['cv_iter'] = cv_iter
            # create a subdirectory under the same model name
            all_settings['logger']['name'] = f"{logger_name}/cv_iter_{cv_iter}"
            train_model(all_settings)
    else:
        train_model(all_settings)
