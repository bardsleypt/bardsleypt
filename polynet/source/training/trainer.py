import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.callbacks import TensorBoard
from source.training.utils import get_optimizer, get_loss_fct, get_model_factory
from source.models.mvn import MVN
from source.data.generator import DataGenerator
from source.data.utils import load_data
from source.callbacks.metrics import EvaluateCallback, PrecRecCallback
from source.callbacks.tb_visuals import TBPlotter
from source.callbacks.utils import gen_checkpoint_callback
from source.utils.logger import Logger

logger = Logger().get_logger()


class Trainer(object):
    def __init__(self, config):
        # Set seed for training
        seed = config['classif_opts']['seed']
        logger.info('Setting random seed: {}'.format(seed))
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Create checkpoint directory
        if not os.path.exists(config['global_opts']['checkpoint_dir']):
            os.makedirs(config['global_opts']['checkpoint_dir'])
        self.config = config
        self.model_factory = None

        # setup model factory and model
        model_name = self.config['classif_opts']['model']
        model_params = self.config['classif_opts']['model_params'][model_name]
        logger.info('Setting model: {}'.format(model_name))
        self.model_factory = get_model_factory(model_name, model_params)
        self.model = self.model_factory.create_model()

        # setup optimizer
        optimizer_name = self.config['classif_opts']['optimizer']
        optimizer_params = self.config['classif_opts']['optimizer_params'][optimizer_name]
        logger.info('Setting optimizer: {}'.format(optimizer_name))
        self.optimizer = get_optimizer(optimizer_name, optimizer_params)

        # setup loss function
        loss_fct = self.config['classif_opts']['loss_fct']
        logger.info('Setting loss function: {}'.format(loss_fct))
        self.loss_fct = get_loss_fct(loss_fct)

        # compile model
        self.model.compile(self.optimizer, self.loss_fct)
        self.model.summary(print_fn=logger.info)

        # load data sets
        for ds_name, ds_config in self.config['classif_opts']['datasets'].items():
            df_name = '{}_df'.format(ds_name)
            logger.info('loading dataset {}'.format(df_name))
            data = load_data(**ds_config)
            setattr(self, df_name, data)

        # setup data generator for training
        data = pd.DataFrame()
        for ds_name in self.config['classif_opts']['train_datasets']:
            data = pd.concat([data, getattr(self, '{}_df'.format(ds_name))], axis=0). \
                reset_index(drop=True)
        logger.info('Instantiating training DataGenerator')
        self.train_gen = DataGenerator(data,
                                       batch_size=self.config['classif_opts']['batch_size'],
                                       shuffle=True,
                                       keep_last_batch=False)

        # setup data validation generators used for metrics tracking and checkpointing
        gens = dict()
        for ds_name in self.config['classif_opts']['valid_datasets']:
            data = getattr(self, '{}_df'.format(ds_name))
            logger.info('Instantiating validation DataGenerator: {}'.format(ds_name))
            gens[ds_name] = DataGenerator(data,
                                          batch_size=self.config['classif_opts']['batch_size'],
                                          shuffle=False,
                                          keep_last_batch=True)
        self.valid_gens = gens

        # Setup callbacks for loss metrics and model tracking/checkpointing
        self.callbacks = []
        self.callbacks.append(EvaluateCallback(self.valid_gens, verbose=True))
        self.callbacks.extend(
            [gen_checkpoint_callback(self.config['global_opts']['checkpoint_dir'],
                                     monitor='{}_loss'.format(gen_name))
             for gen_name in self.valid_gens.keys()])

        # Setup callbacks prec/recall metrics and model tracking/checkpointing
        self.callbacks.append(PrecRecCallback(self.valid_gens, verbose=True))
        self.callbacks.extend(
            [gen_checkpoint_callback(self.config['global_opts']['checkpoint_dir'],
                                     monitor='{}_{}'.format(gen_name, met))
             for gen_name in self.valid_gens.keys() for met in ['prec', 'rec']])

        # Setup TensorBoard callbacks if tb_log_dir specified
        tb_log_dir = self.config['global_opts'].get('tb_log_dir')
        if tb_log_dir:
            self.callbacks.append(TBPlotter('{}/{}'.format(tb_log_dir, 'images')))
            self.callbacks.append(
                TensorBoard(log_dir=tb_log_dir, update_freq='epoch', histogram_freq=1))

    def fit(self):
        if any(map(lambda lay: isinstance(lay, MVN), self.model.layers)):
            logger.info('Updating mvn')
            self.model.evaluate(self.train_gen, verbose=1)
            for layer in self.model.layers:
                if isinstance(layer, MVN):
                    logger.info(
                        'MVN updated with {} samples, '
                        'disabling updates for layer {}'.format(layer.num_samples.numpy(),
                                                                layer.name))
                    w = layer.get_weights()
                    # weights are returned in order so w[0] is mnv update flag,
                    # we set it here to False
                    w[0] = tf.constant(False)
                    layer.set_weights(w)

        hist = self.model.fit(self.train_gen,
                              steps_per_epoch=len(self.train_gen),
                              epochs=self.config['classif_opts']['num_epochs'],
                              verbose=1,
                              callbacks=self.callbacks,  # Validation in callbacks
                              validation_data=None,
                              validation_freq=1,
                              class_weight=None,
                              max_queue_size=10,
                              )
        return hist, self
