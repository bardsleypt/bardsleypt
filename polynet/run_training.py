import argparse
import os
import sys
import yaml
from collections import OrderedDict
from source.utils.logger import Logger, log_dict
from source.utils.wrappers import setup_data, train_model, calibrate_model, test_model
logger = Logger().get_logger()


def validate_config(conf):
    if 'data_opts' in conf:
        # Append data_dir to paths
        data_dir = conf['global_opts']['data_dir']
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for ds_name, ds_config in conf['data_opts']['datasets'].items():
            if os.path.dirname(ds_config['filename']) != data_dir:
                conf['data_opts']['datasets'][ds_name]['filename'] = \
                    os.path.join(data_dir, ds_config['filename'])

    if 'classif_opts' in conf:
        assert 'datasets' in conf['classif_opts'], 'datasets for train and valid must be ' \
                                               'specified in classif_opts'
        # Append data_dir to paths
        data_dir = conf['global_opts']['data_dir']
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for ds_name, ds_config, in conf['classif_opts']['datasets'].items():
            if os.path.dirname(ds_config['filename']) != data_dir:
                conf['classif_opts']['datasets'][ds_name]['filename'] = \
                    os.path.join(data_dir, ds_config['filename'])

    if 'calib_opts' in conf:
        assert 'datasets' in conf['classif_opts'], 'datasets for calibration must be ' \
                                               'specified in calib_opts'
        # Append data_dir to paths
        data_dir = conf['global_opts']['data_dir']
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for ds_name, ds_config, in conf['calib_opts']['datasets'].items():
            if os.path.dirname(ds_config['filename']) != data_dir:
                conf['calib_opts']['datasets'][ds_name]['filename'] = \
                    os.path.join(data_dir, ds_config['filename'])

    return conf


def main(argv):
    """
    Runs feature extraction as well ad DNN training
    :param argv: optional
    :return: None
    """
    if argv is not None:
        argv = argv[1:]

    # Parse arguments
    parser = argparse.ArgumentParser(description='Automated asset generation')
    parser.add_argument('--step',
                        choices=['setup_data',
                                 'train_model',
                                 'calibrate_model',
                                 'test_model',
                                 'all'],
                        required=False,
                        default='setup_data',
                        help='Different steps for creating the assets. Default is %('
                             'default)s)')
    parser.add_argument('--only',
                        dest='this_step_only',
                        action='store_true',
                        help='runs only the given step')
    parser.add_argument('--config',
                        dest='config',
                        default='training_config.yaml',
                        help='training config')

    args = vars(parser.parse_args(argv))
    with open(os.path.abspath(args['config'])) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Setup logger
    logger = Logger(output_dir=os.path.join(config['global_opts']['results_dir'], 'logs'),
                    c_level=config['logging_opts']['console_level'],
                    f_level=config['logging_opts']['file_level'])

    logger.info('Validating config')
    config = validate_config(config)
    log_dict(config)

    # Setup function calls for each step
    steps = OrderedDict((
        ('setup_data', lambda: setup_data(config)),
        ('train_model', lambda: train_model(config)),
        ('calibrate_model', lambda: calibrate_model(config)),
        ('test_model', lambda: test_model(config))
    ))

    # Main loop to iterate through step
    if args['step'] == 'all':
        index = 0
    else:
        index = list(steps.keys()).index(args['step'])
    try:
        for i, (step, fct) in enumerate(steps.items()):
            if i >= index:
                logger.info('Running step: {}'.format(step))
                fct()
                if args['this_step_only']:
                    break
            else:
                logger.info('Skipping step: {}'.format(step))
        return 0

    except Exception as e:
        logger.error(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
