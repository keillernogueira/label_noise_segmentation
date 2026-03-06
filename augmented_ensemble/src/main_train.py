import json
import logging
import os
from pathlib import Path

import lightning as pl
import yaml

import numpy as np
import torch
import argparse

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
    tb_logger = pl.pytorch.loggers.TensorBoardLogger(**settings['logger'])
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

    if len(dm.val_fp_imgs) == 0: 
        # Callbacks
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            monitor='train_BinaryF1Score',
            filename='ckpt-{epoch:02d}-{train_BinaryF1Score:.4f}',
            save_top_k=1,
            save_last=True,
            mode='max',
            every_n_epochs=1
        )
    else:
        # Callbacks
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            monitor='val_BinaryF1Score',
            filename='ckpt-{epoch:02d}-{val_BinaryF1Score:.4f}',
            save_top_k=1,
            save_last=True,
            mode='max',
            every_n_epochs=1
        )

    #reg_save_callback = SaveModelAtEpochsCallback(
    #    save_epochs=[10,20,25,30]
    #)
    # save_pred_callback = SaveSegmentationPredictionsCallback(dataloader=dm.train_dataloader, output_dir=os.path.join(tb_logger.log_dir, "predictions"))
    save_pred_callback = SaveSegmentationPredictionsCallback(output_dir=os.path.join(tb_logger.log_dir, "predictions"))

    # Trainer
    trainer_dict = settings['trainer']
    # trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback, summary,save_pred_callback], **trainer_dict)
    trainer = pl.Trainer(logger=None, callbacks=[save_pred_callback], **trainer_dict)

    # Save the config file, after adding the slurm job id (if exists)
    path_out = Path(tb_logger.log_dir, 'settings.yaml')
    setting_dict_save = settings.copy()
    setting_dict_save['SLURM_JOBID'] = os.environ['SLURM_JOBID'] if 'SLURM_JOBID' in os.environ else None
    with open(path_out, 'w') as fp:
        yaml.dump(setting_dict_save, fp, sort_keys=False)
    logger.info(f'Settings saved to to {path_out}')
    logger.info(f'Exported settings:\n{json.dumps(settings, sort_keys=False, indent=4)}')

    trainer.fit(task, dm)
    #logger.info(f'Best model {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}')

# ---------- helper ----------

class SaveSegmentationPredictionsCallback(pl.Callback):
    def __init__(self, output_dir="seg_predictions", max_batches=999999):
        super().__init__()
        self.output_dir = output_dir
        self.max_batches = max_batches
        os.makedirs(self.output_dir, exist_ok=True)
        self.initialized = False
        self.total_images = 5000
        self.filenames = []
        self.labels = []

    def _generate_and_store_predictions(self, pl_module, dataloader, epoch=0):
        pl_module.eval()
        device = pl_module.device
        
        saved_predictions = []
        saved_labels = []
        saved_names = []
        saved_idx = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print(f"Epoch {epoch} - Store {i+1} of {len(dataloader)} .... ")
                print(f"Epoch {epoch} | Batch {i+1}/{len(dataloader)}", end="\r")

                batch["img"] = batch["img"].to(device)
                probs = pl_module(batch)
                probs_cpu = probs.detach().cpu()
                saved_predictions.append((probs_cpu * 255).byte().numpy())

                # clear GPU
                del probs, probs_cpu
                torch.cuda.empty_cache()

                saved_labels.append(batch["mask"].cpu().bool().numpy())
                saved_names += batch["name"]
                saved_idx += batch["idx"]
                batch["img"] = batch["img"].to("cpu")

                if i + 1 >= self.max_batches:
                    break

            saved_labels = np.concatenate(saved_labels, 0)
            saved_predictions = np.concatenate(saved_predictions, 0)

            print(os.path.join(self.output_dir, "saved_predictions_" +str(epoch).zfill(3)  + ".npz"))
            np.savez_compressed(os.path.join(self.output_dir, "saved_predictions_" +str(epoch).zfill(3)  + ".npz"),
                    preds=saved_predictions,
                    labels=saved_labels,
                    names=saved_names,
                    idx=saved_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        dataloader = trainer.datamodule.val_dataloader()
        current_epoch = trainer.current_epoch
        self._generate_and_store_predictions(pl_module, dataloader, epoch=current_epoch)

class SaveModelAtEpochsCallback(pl.Callback):
    def __init__(self, save_epochs=[10, 20, 30], dirpath=None, filename="ckpt-epoch{epoch:02d}.ckpt"):
        """
        Custom callback to save the model at specific epochs.
        
        Args:
            save_epochs (list): List of epochs at which to save the model.
            dirpath (str, optional): Directory to save checkpoints. Defaults to Lightning logs if None.
            filename (str): File naming format. Use `{epoch}` for epoch-based naming.
        """
        self.save_epochs = set(save_epochs)
        self.dirpath = dirpath  # Custom save directory (default: None)
        self.filename = filename  # Naming pattern for checkpoints

    def on_train_epoch_start(self, trainer, pl_module):
        return super().on_train_epoch_start(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of every epoch"""

        if trainer.current_epoch in self.save_epochs:
            # Set directory path (default: trainer log directory)
            save_dir = self.dirpath or os.path.join(trainer.log_dir, "checkpoints")

            # Ensure directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Format filename
            save_path = os.path.join(save_dir, self.filename.format(epoch=trainer.current_epoch))

            # Save model checkpoint
            trainer.save_checkpoint(save_path)
            print(f"Checkpoint saved at epoch {trainer.current_epoch}: {save_path}")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description='Train segmentation model (supports list of seeds in config)')
    parser.add_argument('--settings', type=str, default='configs/default_settings.yaml', help='Path to YAML settings file')
    args = parser.parse_args()

    settings_fp = args.settings
    with open(settings_fp, 'r') as fp:
        all_settings = yaml.load(fp, Loader=yaml.FullLoader)

    # normalize seeds to a list
    seed_list = all_settings['task']['seed'] if isinstance(all_settings['task']['seed'], list) else [all_settings['task']['seed']]

    # set the number of workers to 0 if the data is loaded in memory
    if all_settings['data'].get('in_memory', False):
        print('Setting the number of workers to 0')
        all_settings['data']['num_workers'] = 0

    logger_name = all_settings['logger']['name']

    # iterate CV folds and seeds; the settings file controls cv_n_splits and seeds
    for cv_iter in range(all_settings['data'].get('cv_n_splits', 1)):
        for seed in seed_list:
            cfg = all_settings.copy()
            cfg['task'] = cfg.get('task', {}).copy()
            cfg['data'] = cfg.get('data', {}).copy()

            cfg['task']['seed'] = seed
            cfg['data']['cv_iter'] = cv_iter

            # per-run logger name (keeps original base name)
            cfg['logger'] = cfg.get('logger', {}).copy()
            cfg['logger']['name'] = f"{logger_name}/cv_iter_{cv_iter}_seed_{seed}"

            train_model(cfg)
