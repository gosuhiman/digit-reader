import math
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


def get_train_samples(count=2000):
    train_labels = []
    train_samples = []

    for i in range(math.ceil(count * 0.02)):
        train_samples.append(randint(13, 64))
        train_labels.append(1)

        train_samples.append(randint(65, 100))
        train_labels.append(0)

    for i in range(math.ceil(count * 0.48)):
        train_samples.append(randint(13, 64))
        train_labels.append(0)

        train_samples.append(randint(65, 100))
        train_labels.append(1)

    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    train_samples, train_labels = shuffle(train_samples, train_labels)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

    return scaled_train_samples, train_labels
