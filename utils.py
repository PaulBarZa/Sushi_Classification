from tensorflow.keras.preprocessing import image_dataset_from_directory 
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def load_data(shape=(160, 160), dir='./dataset/'):
    ds = image_dataset_from_directory(
        dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        shuffle=False,
        image_size=shape,
        color_mode='rgb',
        interpolation='bilinear'
    )
    # X = 1927 images de tailles 100x100x3
    # y = 1927 labels avec 3 valeurs [bug, mouse, screen] -> ex : [0, 0, 1]
    X = np.concatenate([images.numpy() for images, _ in ds])
    y = np.concatenate([labels.numpy() for _, labels in ds])
    idx = list(range(len(X)))
    random.shuffle(idx)
    return X[idx], y[idx]


# Custom data augmentation
def data_augmentation(X, y, augmentation_nb=0):
    # augmentation_nb: how many augmentation you want to do

    augmented_X_1, augmented_X_2, augmented_X_3, augmented_X_4, augmented_X_5, augmented_X_6 = np.split(X, 6)
    y_1, y_2, y_3, y_4, y_5, y_6 = np.split(y, 6)
    
    if(augmentation_nb > 0):
        #Rotate and flip
        random_rotateflip = keras.Sequential([
            layers.experimental.preprocessing.RandomRotation((-0.4, 0.4)),
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
        ])
        random_rotateflip_X = random_rotateflip(augmented_X_1)
        X = np.concatenate((X, random_rotateflip_X))
        y = np.concatenate((y, y_1))

    if(augmentation_nb > 1):
        #Zomm
        random_zoom = keras.Sequential([
            layers.experimental.preprocessing.RandomZoom(.5, .5),
        ])
        zommed_X = random_zoom(augmented_X_2)
        X = np.concatenate((X, zommed_X))
        y = np.concatenate((y, y_2))

    if(augmentation_nb > 2):
        #contrast
        random_contrast = keras.Sequential([
            layers.experimental.preprocessing.RandomContrast(0.2)
        ])
        contrasted_X = random_contrast(augmented_X_3)
        X = np.concatenate((X, contrasted_X))
        y = np.concatenate((y, y_3))

    if(augmentation_nb > 3):
        #Rotate
        random_rotate = keras.Sequential([
            layers.experimental.preprocessing.RandomRotation((-0.4, 0.4)),
        ])
        random_rotate_X = random_rotate(augmented_X_4)
        X = np.concatenate((X, random_rotate_X))
        y = np.concatenate((y, y_4))

    if(augmentation_nb > 4):
        #flip
        random_flip = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
        ])
        random_flip_X = random_flip(augmented_X_5)
        X = np.concatenate((X, random_flip_X))
        y = np.concatenate((y, y_5))


    return X, y
