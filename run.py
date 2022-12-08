import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from tensorflow import keras
from tensorflow.keras import layers

import io
import imageio
import time

from tools import get_image
from model import Fit

lr_image_shape = [180, 320, 3]
image_shape = [720, 1280, 3]
window_size = 5
train_y_dir = "/home/niteya/work/ML/train/train_sharp/"
train_x_dir = "/home/niteya/work/ML/train/train_sharp_bicubic/"
batch_div = 48
iterations = 100

dataset = tf.data.Dataset.from_generator(
    get_image,
    output_signature=(
        tf.TensorSpec(shape=(None, window_size, *lr_image_shape), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *image_shape), dtype=tf.float32),
    ),
    args=(window_size, lr_image_shape, train_x_dir, train_y_dir, batch_div),
)

dataset = dataset.prefetch(tf.data.AUTOTUNE)

fit = Fit(lr_image_shape, image_shape)

for i in range(100):
    start = time.time()
    for step, (input_image, target) in dataset.enumerate():
        if (step) % 100 == 0:

            if step != 0:
                print(f"Time taken for 100 steps: {time.time()-start:.2f} sec\n")
            start = time.time()

        fit.train_step(input_image, target, step + i * batch_div * 240)

        # Training step
        if (step + 1) % 10 == 0:
            print(".", end="", flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 300 == 0:
            fit.checkpoint.save(file_prefix=fit.checkpoint_prefix)
