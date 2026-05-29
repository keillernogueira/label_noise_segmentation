import glob
import numpy as np
import os
import rasterio
import tqdm
import random


def compute_norm_std():
    """
    Computes the mean, standard deviation, maximum, and minimum values for a set of raster images.

    Parameters:
    split_file_path (str): Path to the CSV file containing the list of image files.
    data_root_path (str): Root directory where the image files are stored.

    Returns:
    tuple: A tuple containing the mean, standard deviation, maximum, and minimum values.
    """

    file_list = glob.glob("data/dataset/training_patches/*")
    random.shuffle(file_list)

    sum = np.array([0.0] * 3)
    sum_sq = np.array([0.0] * 3)
    max_val = np.array([-1.0] * 3)
    min_val = np.array([100000.0] * 3)

    data_list = []
    for i, img in enumerate(tqdm.tqdm(file_list)):
        with rasterio.open(img) as src:
            data = src.read()
            data = np.nan_to_num(data)
        data = data.reshape((3, -1))
        data_list.append(data)
        # sum = sum + np.sum(data, axis=(1,2))
        # sum_sq = sum_sq + np.sum(data * data, axis=(1,2))
        # max_val = np.maximum(np.max(data, axis=(1,2)), max_val)
        # min_val = np.minimum(np.min(data, axis=(1,2)), min_val)


        if i % 10 == 0:
            data_np = np.concatenate(data_list, axis=1)
            std = np.std(data_np, axis=1)
            mean = np.mean(data_np, axis=1)
            max_val = np.max(data_np, axis=1)
            min_val = np.min(data_np, axis=1)
            # n = (i + 1.0) * 640.0 * 640.0
            # mean = sum / n
            # # tmp = sum_sq / n - mean * mean
            # # var = tmp / (n - 1)
            # var = ((n * sum_sq) - (sum * sum)) / (n * (n - 1))
            # print(var)
            # std = np.sqrt(var)
        
            print("Mean, std: ", mean, std)
            print("Max values:", max_val)
            print("Min values:", min_val)

    return mean, std, max_val, min_val

# Example usage
mean, std, max_val, min_val = compute_norm_std()

print("Mean, std: ", mean, std)
print("Max values:", max_val)
print("Min values:", min_val)



