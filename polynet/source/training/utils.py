from keras.optimizer_v2.adam import Adam
from keras.losses import CategoricalCrossentropy
from source.models.mlpnet import MLPNet
from source.models.derivnet import DerivNet
from source.utils.logger import Logger

logger = Logger().get_logger()


def get_optimizer(name, params):
    if name.lower() == 'adam':
        return Adam(**params, name=name)
    else:
        msg = "Error: optimizer {} is not defined".format(name)
        logger.error(msg)
        raise Exception(msg)


def get_loss_fct(name):
    if name.lower() in ['crossentropy', 'crossent']:
        loss = CategoricalCrossentropy(name='CrossEntropyLoss',
                                       from_logits=True,
                                       label_smoothing=0,
                                       axis=-1)
        return loss
    else:
        msg = "Error: loss_fct {} is not defined".format(name)
        logger.error(msg)
        raise Exception(msg)


def get_model_factory(name, params):
    if name.lower() == 'derivnet':
        return DerivNet(**params)
    if name.lower() == 'mlpnet':
        return MLPNet(**params)
    else:
        msg = "Error: model {} is not defined".format(name)
        logger.error(msg)
        raise Exception(msg)
