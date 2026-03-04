import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn
import torchmetrics as tm
from PIL import Image
from building_footprint_segmentation.helpers.normalizer import min_max_image_net

from utils.data import get_augmentations, get_reveresed_augmentations


class AleatoricUQLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, target):
        y_mean_sigmoid = input[:, 0, ...]
        y_logvar = input[:, 1, ...]
        bce_loss = torch.nn.functional.binary_cross_entropy(y_mean_sigmoid, target.squeeze(dim=1))
        loss = self.alpha * bce_loss * torch.exp(-y_logvar) + (1 - self.alpha) * y_logvar

        return loss.mean()


class SegTask(pl.LightningModule):
    def __init__(self, model, task_params, outdir=None):
        super().__init__()

        self.model = model
        self.thr = 0.5
        self.val_metrics = tm.MetricCollection([
            tm.Accuracy(threshold=self.thr, task='binary'),
            tm.JaccardIndex(threshold=self.thr, task='binary'),
            tm.Precision(threshold=self.thr, task='binary'),
            tm.Recall(threshold=self.thr, task='binary'),
            tm.F1Score(threshold=self.thr, task='binary')
        ])
        self.optimizer_settings = task_params['optimization']['optimizer']
        self.lr_scheduler_settings = task_params['optimization']['lr_schedule']

        # get the main logger
        self._logger = logging.getLogger('pytorch_lightning.core')
        self.outdir = outdir

        # initialize the train/val metrics accumulators
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # check whether the model predicts the aleatoric uncertainty
        loss_name = task_params['loss']['name']
        self.is_aleatoric_uq_model = (loss_name == 'AleatoricUQLoss')

        # loss function
        if self.is_aleatoric_uq_model:
            loss_class = AleatoricUQLoss
        else:
            assert hasattr(torch.nn, loss_name), f'Loss {loss_name} not found in torch.nn'
            loss_class = getattr(torch.nn, loss_name)
        self.loss = loss_class(**task_params['loss']['args'] if task_params['loss']['args'] is not None else {})

        self.use_tta = task_params['use_tta']
        self.n_samples_tta = task_params['n_samples_tta'] if self.use_tta else 1
        self.augmentations = get_augmentations() if self.use_tta else None

    def forward(self, batch):
        images = batch['img']

        # batch, channels, height, width
        images = images.permute(0, 3, 1, 2)

        out = self.model(images)

        # if the model predicts the aleatoric uncertainty, apply the sigmoid function on the mean part
        if self.is_aleatoric_uq_model:
            out[:, 0, ...] = torch.sigmoid(out[:, 0, ...])

        return out

    def configure_optimizers(self):
        optimizers = [
            getattr(torch.optim, o['name'])(self.parameters(), **o['args'])
            for o in self.optimizer_settings
        ]
        if self.lr_scheduler_settings is None:
            return optimizers
        schedulers = [
            getattr(torch.optim.lr_scheduler, s['name'])(optimizers[i], **s['args'])
            for i, s in enumerate(self.lr_scheduler_settings)
        ]
        return optimizers, schedulers

    def aggregate_step_metrics(self, step_outputs, label):
        # aggregate the loss and the metrics
        metrics = {
            f"{label}_{k}": torch.stack([x[k] for x in step_outputs]).mean()
            for k in step_outputs[0].keys()
        }

        return metrics

    def training_step(self, batch, batch_idx):
        out = self(batch)
        mask = batch['mask']
        loss = self.loss(input=out, target=mask.unsqueeze(dim=1))

        tb_logs = {'loss': loss}
        self.log_dict(tb_logs, on_epoch=False, on_step=True, batch_size=len(out), sync_dist=True)

        # compute the evaluation metrics for each element in the batch
        metrics = self.val_metrics(out[:, 0, ...], mask)
        metrics.update(tb_logs)

        self.training_step_outputs.append(metrics)

        return metrics

    def on_train_epoch_start(self):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2189
        print('\n')

    def on_train_epoch_end(self):
        avg_tb_logs = self.aggregate_step_metrics(self.training_step_outputs, label='train')

        # show the epoch as the x-coordinate
        avg_tb_logs['step'] = float(self.current_epoch)
        self.log_dict(avg_tb_logs, on_step=False, on_epoch=True, sync_dist=True)

        # clear the accumulator
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        mask = batch['mask']
        loss = self.loss(input=out, target=mask.unsqueeze(dim=1))

        tb_logs = {'loss': loss}
        self.log_dict(tb_logs, on_epoch=True, on_step=True, batch_size=len(out), sync_dist=True)
        metrics = self.val_metrics(out[:, 0, ...], mask)
        metrics.update(tb_logs)

        self.validation_step_outputs.append(metrics)

        return metrics

    def on_validation_epoch_end(self):
        avg_tb_logs = self.aggregate_step_metrics(self.validation_step_outputs, label='val')

        # show the epoch as the x-coordinate
        avg_tb_logs['step'] = float(self.current_epoch)
        self.log_dict(avg_tb_logs, on_step=False, on_epoch=True, sync_dist=True)

        # clear the accumulator
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        out = self(batch)
        out_np = out.detach().cpu().numpy()
        mask = batch['mask']

        loss = self.loss(input=out, target=mask.unsqueeze(dim=1))
        tb_logs = {'loss': loss}
        metrics = self.val_metrics(out[:, 0, ...], mask)
        metrics.update(tb_logs)
        self.test_step_outputs.append(metrics)

        if self.use_tta:
            # loop over the images in the batch
            out_np_list = []
            for fp in batch['fp']:
                img_o = np.array(Image.open(fp))

                # sample multiply augmented images (and save them for reverting later)
                aug_out_list = []
                for _ in range(self.n_samples_tta):
                    aug_out = self.augmentations(image=img_o)
                    aug_out_list.append(aug_out)

                # form a batch with the augmented images (after normalizing them)
                img_aug_batch = np.stack([min_max_image_net(aug_out['image']) for aug_out in aug_out_list], axis=0)
                out_aug = self({'img': torch.tensor(img_aug_batch).to(self.device)})

                # revert the augmentations
                out_aug_np = out_aug.detach().cpu().numpy()
                img_aug_rev_list = []
                for i, aug_out in enumerate(aug_out_list):
                    rev_transform = get_reveresed_augmentations(aug_out)
                    img_aug_rev_list.append(rev_transform(image=out_aug_np[i, 0, ...])['image'])

                out_np_list.append(np.mean(img_aug_rev_list, axis=0))

            out_np = np.stack(out_np_list, axis=0)

            # add back the channel dimension
            out_np = np.expand_dims(out_np, axis=1)

        # export the predictions
        for i, fp in enumerate(batch['fp']):
            pred = out_np[i, 0, ...]
            pred = (pred > self.thr).astype(np.uint8) * 255
            pred = Image.fromarray(pred)
            fp_pred = Path(self.outdir, Path(fp).name)
            fp_pred.parent.mkdir(parents=True, exist_ok=True)
            pred.save(fp_pred)

            # export the aleatoric uncertainty if the model predicts it
            if self.is_aleatoric_uq_model:
                uq = out_np[i, 1, ...]
                uq = (uq - uq.min()) / (uq.max() - uq.min()) * 255
                uq = Image.fromarray(uq.astype(np.uint8))
                fp_uq = Path(self.outdir.parent / 'preds_uq' / fp_pred.name)
                fp_uq.parent.mkdir(parents=True, exist_ok=True)
                uq.save(fp_uq)

    def on_test_epoch_end(self):
        avg_tb_logs = self.aggregate_step_metrics(self.test_step_outputs, label='test')

        print("\nTest results:")
        nc = max([len(k) for k in avg_tb_logs.keys()])
        for k, v in avg_tb_logs.items():
            print(f"\t{k:{nc}}: {float(v.cpu().numpy()):.4f}")
        print()

        # clear the accumulator
        self.test_step_outputs.clear()
