#!/usr/bin/env python3
"""
Minimal ensemble averaging script.

Usage:
  python ensemble_average.py --inputs run1_dir run2_dir ... --epoch 10 --out ensemble_epoch_010.npz --threshold 0.5 --save-masks

Each input can be either a path to a directory containing `saved_predictions_XXX.npz`
or a direct path to an NPZ file. The script loads `preds` arrays, averages them
in floating range [0,1], writes an ensemble NPZ and optionally thresholded PNGs.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import os


def load_preds(path, epoch):
    p = Path(path)
    if p.is_file() and p.suffix == '.npz':
        data = np.load(p, allow_pickle=True)
        preds = data['preds']
        names = data.get('names', None)
        return preds, names

    # assume directory containing saved_predictions_{epoch:03d}.npz
    npz = p / f"saved_predictions_{epoch:03d}.npz"
    if npz.exists():
        data = np.load(npz, allow_pickle=True)
        preds = data['preds']
        names = data.get('names', None)
        return preds, names

    raise FileNotFoundError(f'No NPZ predictions found at {path} for epoch {epoch}')


def main():
    parser = argparse.ArgumentParser(description='Average per-seed predictions into an ensemble')
    parser.add_argument('--inputs', nargs='+', required=True, help='Paths to run dirs or NPZ files')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number to load (integer)')
    parser.add_argument('--out', type=str, required=True, help='Output NPZ path for ensemble predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold to binarize ensemble')
    parser.add_argument('--save-masks', action='store_true', help='Also save thresholded masks as PNGs in out_dir/masks')
    args = parser.parse_args()

    all_preds = []
    names = None

    for inp in args.inputs:
        preds, n = load_preds(inp, args.epoch)
        # preds may be uint8 [0,255] or float; convert to float [0,1]
        preds = preds.astype('float32')
        if preds.max() > 1.0:
            preds = preds / 255.0
        all_preds.append(preds)
        if names is None and n is not None:
            names = list(n)

    stacked = np.stack(all_preds, axis=0)  # (n_models, N, ...)
    ensemble = np.mean(stacked, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save ensemble soft predictions as float32
    if names is not None:
        np.savez_compressed(out_path, preds=ensemble.astype('float32'), names=np.array(names, dtype='S'))
    else:
        np.savez_compressed(out_path, preds=ensemble.astype('float32'))

    print(f'Ensemble saved to {out_path}')

    if args.save_masks:
        masks_dir = out_path.parent / (out_path.stem + '_masks')
        masks_dir.mkdir(parents=True, exist_ok=True)

        # ensemble shape e.g. (N,1,H,W) or (N,H,W)
        arr = ensemble
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]

        for i in range(arr.shape[0]):
            mask = (arr[i] >= args.threshold).astype('uint8') * 255
            fname = names[i].decode() if names is not None else f'mask_{i:04d}.png'
            Image.fromarray(mask).save(masks_dir / fname)

        print(f'Thresholded masks saved to {masks_dir}')


if __name__ == '__main__':
    main()
