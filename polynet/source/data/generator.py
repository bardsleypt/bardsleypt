import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from source.utils.logger import Logger

logger = Logger().get_logger()


class DataGenerator(keras.utils.data_utils.Sequence):
    def __init__(self, dataframe,
                 feat_key='features',
                 label_key='label',
                 batch_size=32,
                 shuffle=False,
                 keep_last_batch=True):
        self.batch_size = batch_size
        self.keep_last_batch = keep_last_batch  # keep the last batch or not
        self.shuffle = shuffle
        self.features = np.vstack(dataframe[feat_key])
        self.labels = np.vstack(dataframe[label_key])
        self.index = np.arange(len(self.labels))
        self.on_epoch_end()

    def __get_data(self, index):
        feats = self.features[index, :]
        labels = self.labels[index]
        return feats, np.asarray(labels).astype(int)

    def __len__(self):
        if len(self.labels) % self.batch_size == 0:
            return int(np.floor(len(self.labels) / self.batch_size))
        else:
            return int(np.floor(len(self.labels) / self.batch_size)) \
                   + self.keep_last_batch

    def __getitem__(self, index):
        # Get index for batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]

        # Get data from index
        x, y = self.__get_data(index)
        return x, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.index)
