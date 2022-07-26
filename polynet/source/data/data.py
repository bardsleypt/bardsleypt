import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from source.utils.logger import Logger

logger = Logger().get_logger()


def get_data_factory(data_type, name, **ds_config):
    if data_type.lower() == 'synthetic':
        return ShapeDataGen(name, **ds_config)
    elif data_type.lower() == 'csv':
        # TODO: implement csv reader for csv data types
        pass
    elif data_type.lower() == 'web':
        # TODO: implement web reader for web data types
        pass
    else:
        msg = 'Unknown dataset type encountered in get_data_factory'
        logger.error(msg)
        raise Exception(msg)
    pass


class BaseDataFactory(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup_data(self, *args, **kwargs):
        pass


class ShapeDataGen(BaseDataFactory):
    def __init__(self, name, **ds_config):
        self.name = name
        self.dim = ds_config['dim']
        self.filename = ds_config['filename']
        self.subset_configs = dict()
        for conf_key, conf_val in ds_config.items():
            if 'subset' in conf_key.lower():
                self.subset_configs[conf_key] = self.validate_config(conf_val)
        if len(self.subset_configs) == 0:
            msg = 'Warning, no data subsets found for dataset {}'.format(name)
            logger.warn(msg)

    @staticmethod
    def validate_config(config):
        if any([x not in config.keys() for x in ['num_obs', 'label', 'x_range', 'roots']]):
            msg = 'num_obs, label, x_range, and roots must be specified for ShapeDataGen'
            logger.error(msg)
            raise Exception(msg)
        return config

    def setup_data(self, *args, **kwargs):
        feats, labels = np.empty((0, self.dim), dtype=float), np.empty((0,))
        for name, config in self.subset_configs.items():
            feats_, labels_ = self.gen_base_shape_labels(config['x_range'], config['roots'],
                                                         config['label'], config['num_obs'])
            feats_ = self.apply_shifts(feats_, config.get('shift'))
            feats_ = self.apply_noise(feats_, **config.get('noise'), name=name)
            feats, labels = np.append(feats, feats_, axis=0), np.append(labels, labels_, axis=0)

        df = pd.DataFrame({'features': feats.tolist(), 'label': labels.tolist()})
        df.to_pickle(self.filename, compression='infer')

    def gen_base_shape_labels(self, x_range, roots, label, num_obs):
        x = np.linspace(*x_range, self.dim).reshape(1, -1)
        feats = 1
        for root in roots:
            feats *= (x - root)
        feats = np.tile(feats, (num_obs, 1)).squeeze()
        labels = np.tile(label, (num_obs, 1)).squeeze()
        return feats, labels

    @staticmethod
    def apply_shifts(feats, shift=None):
        if shift:
            shifts = shift[0] + (shift[1] - shift[0]) * np.random.rand(feats.shape[0], 1)
            feats += shifts
        return feats

    def apply_noise(self, feats, dist=None, params=(0, 1), **kwargs):
        if dist:
            if dist.lower() == 'uniform':
                a, b = params
                noise = a + (b - a) * np.random.rand(*feats.shape)
            elif dist.lower() in ['gaussian', 'normal']:
                mu, sigma = params
                noise = mu + sigma * np.random.randn(*feats.shape)
            else:
                msg = 'Invalid noise distribution supplied in {}:{}\n'\
                      'No noise will be added'.format(self.name, kwargs.get('name'))
                logger.warn(msg)
                noise = 0
            feats += noise
        return feats
