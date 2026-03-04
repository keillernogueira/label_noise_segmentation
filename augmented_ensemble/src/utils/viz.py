import cv2
import numpy as np


def draw_contours_from_mask(img, mask, color, thickness, alpha=1.0):
    """
        Extract contours from mask and draw them on the image.
    """
    img_cp = img.astype(np.float32).copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    img_with_contours = cv2.drawContours(
        image=img_cp.copy(),
        contours=contours,
        contourIdx=-1,
        color=color,
        thickness=thickness,
    )

    img_out = cv2.addWeighted(img_cp, 1 - alpha, img_with_contours, alpha, 0)

    return img_out


def contrast_stretch(img, q_lims_clip=(0, 1), per_band=False):
    # clip the values  and scale to [0, 1], per band
    # add a new dimension for grayscale (will remove it at the end)

    img = img.astype(np.float32)
    if len(img.shape) == 2:
        img = img[:, :, None]

    k_list = list(range(img.shape[-1])) if per_band else [None]
    for k in k_list:
        # compute the limits
        _min = np.nanquantile(img[:, :, k], q_lims_clip[0])
        _max = np.nanquantile(img[:, :, k], q_lims_clip[1])

        if _min == _max:
            continue

        # clip
        img[:, :, k][img[:, :, k] < _min] = _min
        img[:, :, k][img[:, :, k] > _max] = _max

        # scale to [0, 1]
        img[:, :, k] = (img[:, :, k] - _min) / (_max - _min)
        img[:, :, k][img[:, :, k] > 1] = 1.0  # rounding errors

    # remove the dummy dimension
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    return img


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open('../../data/external/dataset/training_patches/0_20_66_627_0.png')
    img = np.array(img)
    plt.figure(dpi=200)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')

    img_c = contrast_stretch(img, q_lims_clip=(0.025, 0.975), per_band=False)

    plt.subplot(1, 2, 2)
    plt.imshow(img_c)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
