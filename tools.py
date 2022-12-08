import os

import imageio
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def get_image(
    window_size=5,
    image_shape=(180, 320, 3),
    load_directory_x="val_sharp_bicubic/",
    load_directory_y="val_sharp/",
    batch_div=8,
):
    load_directory_x = load_directory_x.decode(encoding="utf-8")
    load_directory_y = load_directory_y.decode(encoding="utf-8")
    for directories in list(os.listdir(load_directory_x)):
        windows_lr = []
        windows_hr = []
        for images in os.listdir(load_directory_x + directories):
            windows_lr.append(
                np.asarray(
                    imageio.imread(load_directory_x + directories + "/" + images)
                )
            )
        for images in os.listdir(load_directory_y + directories):
            windows_hr.append(
                np.asarray(
                    imageio.imread(load_directory_y + directories + "/" + images)
                )
            )
        train_x = np.array(windows_lr, dtype=np.float32) / 255.0
        train_x = np.squeeze(sliding_window_view(train_x, (5, 180, 320, 3)))
        removed = int(window_size / 2)
        train_y = np.array(windows_hr[removed:-removed], dtype=np.float32) / 255.0
        chunk_size = int(train_x.shape[0]/batch_div)
        for i in range(batch_div):
            yield train_x[i * chunk_size:(i + 1) * chunk_size], train_y[i * chunk_size:(i + 1) * chunk_size]
