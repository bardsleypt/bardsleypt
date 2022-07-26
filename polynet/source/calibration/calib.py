from abc import ABC, abstractmethod
import tensorflow as tf
from keras.layers import Lambda
from source.data.utils import load_data
from source.utils.logger import Logger
from source.models.base_model import BaseModel

logger = Logger().get_logger()


def get_calibrator(method=None, **config):
    params = config.pop('calib_params')
    config.update(**params.get(method, {}))
    if method.lower() in [None, 'nocalib', 'None']:
        return NoCalib(**config)
    elif method.lower() in ['onehot', 'onehotcalib']:
        return OneHotCalib(**config)
    elif method.lower() in ['linconf', 'linconfusion']:
        return LinConfusion(**config)
    else:
        msg = 'Unknown calibration method {}'.format(method)
        logger.error(msg)
        raise Exception(msg)


class BaseCalib(ABC):
    def __init__(self, *args, name, model, datasets=None, **kwargs):
        self.name = name
        self.model = model
        self.datasets = {}
        if datasets is not None:
            for ds_name, ds_config in datasets.items():
                self.datasets[ds_name] = load_data(**ds_config)

    def create_calib_model(self):
        calib_outs = Lambda(self.calibrate_scores, name='calibrate')(self.model.outputs[0])
        calib_model = BaseModel(inputs=self.model.inputs,
                                outputs={'out_raw': self.model.outputs,
                                         'out_cal': calib_outs})
        return calib_model

    @abstractmethod
    def set_calib_mapping(self):
        pass

    @abstractmethod
    def calibrate_scores(self, raw_scores):
        pass


class NoCalib(BaseCalib):
    def __init__(self, *args, name='NoCalib', model, **kwargs):
        super().__init__(name=name, model=model, **kwargs)

    def set_calib_mapping(self):
        pass

    def calibrate_scores(self, raw_scores):
        return raw_scores


class OneHotCalib(BaseCalib):
    def __init__(self, *args, name='OneHotCalib', model, **kwargs):
        super().__init__(name=name, model=model, **kwargs)

    def set_calib_mapping(self):
        pass

    def calibrate_scores(self, raw_scores):
        max_idx = tf.argmax(raw_scores, axis=-1)
        scores = tf.one_hot(max_idx, depth=raw_scores.shape[-1])
        return scores


class PlattCalib(BaseCalib):
    # TODO: Write Platt/LogisticReg calibrator
    # Steps:
    #  1) init with datasets + labels
    #  2) Set weights for classes if desired
    #  3) Use numpy/scikit to find logistic regression weights+bias (y=Ax + b)
    #  4) Set up TF mapping layer using learned weights
    def __init__(self, *args, model, name='PlattCalib', **kwargs):
        super().__init__(name=name, model=model, **kwargs)
        pass

    def set_calib_mapping(self):
        pass

    def calibrate_scores(self, raw_scores):
        pass


class LinConfusion(BaseCalib):
    # TODO: Write linear confusion calibrator
    # Steps:
    #  1) init with datasets + labels + constraints
    #  2) compute all slopes of hyperplanes via pariwise projected log-likelihoods
    #  3) Set each supporting hyperplane constraint with positive leading term
    #  4) Iterate through scores
    #      a) Compute constraint, add 1 to leading term index if positive, else to other term index
    #      b) Sum constrain matrix down rows, find argmax, this gives class membership
    def __init__(self, *args, model, name='LinConfusionCalib', constraints=None, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
        pass

    def set_calib_mapping(self):
        pass

    def calibrate_scores(self, raw_scores):
        pass
