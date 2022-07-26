import pandas as pd
import numpy as np
from source.utils.logger import Logger
from sklearn.preprocessing import OneHotEncoder

logger = Logger().get_logger()


def load_data(filename, feat_key='features', label_key='label',
              categories='auto'):
    logger.info('Loading {}'.format(filename))

    # Load data
    df = pd.read_pickle('{}'.format(filename))
    msg = 'Loaded data: {}'.format(', '.join(['{} ({} class)'.format(val, key)
                                              for key, val in
                                              df[label_key].value_counts().iteritems()]))
    logger.info(msg)

    # Throw away NaNs
    mask = df[feat_key].apply(lambda x: any(np.isnan(x)))
    logger.info('{} features contain nan values'.format(np.sum(mask)))
    df = df[~mask]
    msg = 'After removing nans: {}'.format(', '.join(['{} ({} class)'.format(val, key)
                                                      for key, val in
                                                      df[label_key].value_counts().
                                                      iteritems()]))
    logger.info(msg)

    # Rename columns for easier use in DataGenerator
    df.rename(columns={feat_key: 'features', label_key: 'label'}, inplace=True)
    logger.info('Renamed {{{}: features, {}: label}}'.format(feat_key, label_key))

    # One-hot encode labels
    enc = OneHotEncoder(categories=[categories]).fit(df['label'].to_numpy().reshape(-1, 1))
    labels = enc.transform(df['label'].to_numpy().reshape(-1, 1)).toarray()
    enc_labels = {orig_lab: enc.transform([[orig_lab]]).toarray().squeeze()
                  for orig_lab in enc.categories_[0]}
    msg = 'OneHotEncoding: {}'.format(', '.join(['{} ({} class)'.format(val, key)
                                      for key, val in enc_labels.items()]))
    logger.info(msg)
    df['raw_label'] = df['label']
    df['label'] = labels.tolist()

    df.reset_index(drop=True, inplace=True)
    return df
