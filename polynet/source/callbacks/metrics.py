import numpy as np
from keras.callbacks import Callback
from source.utils.metrics import Metrics
from source.utils.logger import Logger

logger = Logger().get_logger()


class PrecRecCallback(Callback):
    def __init__(self, ds_dict, verbose=False, mode='macro', **kwargs):
        super().__init__(**kwargs)
        self.ds_dict = ds_dict
        self.mode = mode
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        for name, data in self.ds_dict.items():
            msg = ''
            preds = np.argmax(self.model.predict(data), axis=1)
            labels = np.argmax(np.concatenate([lab for _, lab in data]), axis=1)

            prec, rec = Metrics.get_prec_rec(preds, labels, mode=self.mode)

            # Update logs
            logs.update({'{}_prec'.format(name): prec, '{}_rec'.format(name): rec})
            msg += '\t{}_prec: {:.4f}'.format(name, prec)
            msg += '\t{}_rec: {:.4f}'.format(name, rec)

            # if self.verbose, logger.info
            if self.verbose:
                logger.info(msg)
        return logs


class EvaluateCallback(Callback):
    def __init__(self, ds_dict, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.ds_dict = ds_dict
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        msg = ''
        for name, data in self.ds_dict.items():
            key = '_'.join([name, 'loss'])
            loss = self.model.evaluate(data, verbose=False)
            logs.update({key: loss})
            msg += '\t{}: {:.4f}'.format(key, loss)

        if self.verbose:
            logger.info(msg)
        return logs
