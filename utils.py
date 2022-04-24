from tensorflow.keras.preprocessing import image_dataset_from_directory 
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def load_data(shape=(160, 160), dir='./dataset/', color_mode='rgb'):
    ds = image_dataset_from_directory(
        dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        shuffle=False,
        image_size=shape,
        color_mode=color_mode,
        interpolation='bilinear'
    )
    # X = 1927 images de tailles 100x100x3
    # y = 1927 labels avec 3 valeurs [bug, mouse, screen] -> ex : [0, 0, 1]
    X = np.concatenate([images.numpy() for images, _ in ds])
    y = np.concatenate([labels.numpy() for _, labels in ds])
    idx = list(range(len(X)))
    random.shuffle(idx)
    return X[idx], y[idx]
