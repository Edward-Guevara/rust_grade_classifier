import os
import numpy as np


def get_files_with_extension(path, extension):
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(filename)
    return files


def make_folder(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        print(f"The folder {folder_name} already exists.")


def dataset(target, gt, hsi):
    pixel_locations = [np.where(gt == t) for t in target]
    h = [locations[0] for locations in pixel_locations]
    w = [locations[1] for locations in pixel_locations]
    pixels_n = [h.size for h in h]
    dataset = [np.zeros((i, hsi.shape[2])) for i in pixels_n]
    targets = [np.zeros(pixels_n[i]) + target[i] for i in range(len(target))]
    for i in range(len(dataset)):
        for j in range(pixels_n[i]):
            dataset[i][j] = hsi[h[i][j].item(), w[i][j].item(), :]
    data = np.concatenate(tuple(dataset), axis = 0)
    targets = np.concatenate(tuple(targets), axis = 0)        
    return data, targets


# def dataset(target, gt, hsi):
#     pixel_locations = np.where(np.isin(gt, target))
#     pixels_n = [np.count_nonzero(gt == t) for t in target]
#     h, w = pixel_locations
#     dataset = np.zeros((sum(pixels_n), hsi.shape[2]))
#     targets = np.concatenate([np.zeros(n) + t for n, t in zip(pixels_n, target)])
#     dataset = np.take(hsi, np.ravel_multi_index((h, w, np.zeros(len(h), dtype=int)), hsi.shape[:2] + (1,)), axis=2)
#     return dataset, targets