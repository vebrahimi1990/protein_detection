import os
from tifffile import imwrite
from skimage.io import imread
import numpy as np


def make_patches(data_config):
    file_path = data_config['image_dr']
    list_files = os.listdir(file_path)
    M = len(list_files)
    # M = 100
    patch_size = data_config['patch_size']
    threshold = data_config['threshold']

    count = 0
    for k in range(M):
        i = list_files[k]
        f = imread(os.path.join(file_path, '', i)).astype(np.float64)
        w = f.shape[-1]
        h = f.shape[-2]
        if len(f.shape) == 3:
            t = f.shape[0]
        else:
            t = 1
            f = f.reshape((1, f.shape[0], f.shape[1]))
        nx = int(np.ceil(h / patch_size))
        ny = int(np.ceil(w / patch_size))
        count = count + nx * ny * t

    x = np.zeros((count, patch_size, patch_size))
    count1 = 0
    for k in range(M):
        i = list_files[k]
        f = imread(os.path.join(file_path, '', i)).astype(np.float64)
        f = f / f.max()
        f = f - f.min()
        f = f / f.max()
        f[f < 0] = 0
        w = f.shape[-1]
        h = f.shape[-2]
        if len(f.shape) == 3:
            t = f.shape[0]
        else:
            t = 1
            f = f.reshape((1, f.shape[0], f.shape[1]))
        nx = int(np.ceil(h / patch_size))
        ny = int(np.ceil(w / patch_size))
        nnx = np.floor(np.linspace(0, h - patch_size, nx)).astype(np.int32)
        nny = np.floor(np.linspace(0, w - patch_size, ny)).astype(np.int32)
        for j in range(t):
            for p in range(nx):
                for q in range(ny):
                    x[count1, :, :] = f[j, nnx[p]:nnx[p] + patch_size, nny[q]:nny[q] + patch_size]
                    count1 = count1 + 1

    norm_x = np.linalg.norm(x, axis=(1, 2))
    norm_x = norm_x / norm_x.max()
    ind_norm = np.where(norm_x >= threshold)[0]
    y = x[ind_norm]
    y = y / y.max(axis=(-1, -2)).reshape((len(y), 1, 1))
    return y
