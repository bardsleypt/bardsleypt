import os
import datetime
import pandas as pd
import numpy as np
from source.data.data import get_data_factory
from source.data.utils import load_data
from source.training.trainer import Trainer
from source.training.utils import get_model_factory
from source.calibration.calib import get_calibrator
from source.utils.metrics import Metrics
from source.utils.logger import Logger

logger = Logger().get_logger()


def timeit(method):
    def timed(*args, **kw):
        ts = datetime.datetime.now().replace(microsecond=0)
        result = method(*args, **kw)
        te = datetime.datetime.now().replace(microsecond=0)
        logger.info('{} took {}'.format(method.__name__, (te - ts)))
        return result
    return timed


def update_history(train_history, config):
    history = pd.DataFrame(train_history)
    history.index.rename('EPOCH', inplace=True)
    history.to_csv(os.path.join(config['global_opts']['results_dir'],
                                'training_loss.csv'))


def reload_model(model_name, config, model_filepath=None):
    logger.info('(Re)instantiating {} model'.format(model_name))
    if model_name.lower() == 'calib':
        model_name = config['classif_opts']['model']
        model = get_model_factory(model_name,
                                  config['classif_opts']['model_params'][model_name]).create_model()
        model = get_calibrator(model=model, method=config['calib_opts']['method'],
                               calib_params=config['calib_opts']['calib_params']).create_calib_model()
    else:
        model = get_model_factory(model_name,
                                  config['classif_opts']['model_params'][model_name]).create_model()
    if model_filepath is not None:
        logger.info('Loading model weights from {}'.format(model_filepath))
        model.load_weights(model_filepath)
    return model


@timeit
def setup_data(config):
    for name, ds in config['data_opts']['datasets'].items():
        logger.info('Setting up dataset {}'.format(name))
        data_fact = get_data_factory(ds.pop('type'), name, **ds)
        data_fact.setup_data()


@timeit
def train_model(config):
    # Perform training
    hist, trainer = Trainer(config).fit()

    # Update training history csv
    update_history(hist.history, config)


@timeit
def calibrate_model(config):
    # Re-instantiate model
    model_name = config['classif_opts']['model']
    model = reload_model(model_name, config,
                         model_filepath=os.path.join(config['global_opts']['checkpoint_dir'],
                                                     config['calib_opts']['target_model']))

    # Get calibrator
    calib = get_calibrator(model=model, **config['calib_opts'])

    # Perform calibration
    calib.set_calib_mapping()

    # Output calibrated model
    calib_model = calib.create_calib_model()
    calib_model.save_weights(os.path.join(config['global_opts']['checkpoint_dir'],
                                          'cal_model.h5'))


def test_model(config):
    # Re-instantiate model
    model_name = config['test_opts']['model']
    model = reload_model(model_name, config,
                         model_filepath=os.path.join(config['global_opts']['checkpoint_dir'],
                                                     config['test_opts']['target_model']))

    # Load and score datasets
    features, labels, preds, precs, recs = {}, {}, {}, {}, {}
    for ds_name, ds_config in config['test_opts']['datasets'].items():
        df_name = '{}_df'.format(ds_name)
        logger.info('loading dataset {}'.format(df_name))
        features[ds_name], labels[ds_name] = [load_data(**ds_config).get(x)
                                              for x in ['features', 'label']]
        outs = model.predict(np.vstack(features[ds_name]))
        if isinstance(outs, dict):
            preds[ds_name] = outs['out_cal']
        else:
            preds[ds_name] = outs

        # TODO: replace Prec/Rec hard-code with configurable metrics
        precs[ds_name], recs[ds_name] = Metrics.get_prec_rec(preds[ds_name],
                                                             np.vstack(labels[ds_name]))

    metrics_df = pd.DataFrame({'prec': precs, 'rec': recs})
    logger.info('Test Metrics: \n{}'.format(metrics_df))
    metrics_df.to_csv(os.path.join(config['global_opts']['results_dir'],
                                   'test_metrics.csv'))
