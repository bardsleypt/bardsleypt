import logging
import os
import platform
from logging.handlers import RotatingFileHandler
import sys


class Singleton(object):
    """
    Singleton interface:
    """

    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        pass


class Logger(logging.Logger, Singleton):
    c_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)-6s - %(module)-10s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S")

    f_formatter = logging.Formatter(
        fmt='{} %(asctime)s - %(levelname)-8s - %(module)-10s - %(message)s'.
            format(platform.node()))

    def __init__(self, name='model-trainer',
                 output_dir=None,
                 filename='train.log',
                 c_level=logging.INFO,
                 f_level=logging.DEBUG,
                 ):

        super(Logger, self).__init__(name=name)
        self.c_level = c_level
        self.f_level = f_level
        self.filename = filename
        self.output_dir = output_dir

        # Add file handler
        if self.output_dir:
            f_handler = self.file_handler()
            f_handler.name = 'File_Handler'
            f_handler.setLevel(self.f_level)
            f_handler.setFormatter(self.f_formatter)
            self.addHandler(f_handler)

        handler = logging.StreamHandler(sys.stdout)
        handler.name = 'StreamHandler'
        handler.setLevel(self.c_level)
        handler.setFormatter(self.c_formatter)
        self.addHandler(handler)
        self.instance = self

    def file_handler(self):
        # File handler path
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        log_path = os.path.join(self.output_dir, self.filename)
        f_handler = RotatingFileHandler(filename=log_path, maxBytes=1024 * 1024 * 1024,
                                        backupCount=10, delay=False)
        if os.path.isfile(log_path):
            f_handler.doRollover()
        return f_handler

    def get_logger(self):
        return self


logger = Logger().get_logger()


def log_dict(config, depth=0):
    """
    Recursively runs through a dict and logs its content
    :param config: dict
    :param depth: counter
    :return: None
    """
    for (sec, sub_sec) in config.items():
        if isinstance(sub_sec, dict):
            logger.debug('{}{}:'.format(''.join([' ']) * depth, sec))
            log_dict(sub_sec, depth + 5)
        else:
            if not any(filter(lambda x: x in sec, ['type', 'doc'])):
                logger.debug('{}{}: {}'.format(''.join([' ']) * depth, sec, sub_sec))
