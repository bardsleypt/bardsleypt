import os
from keras.callbacks import ModelCheckpoint


def gen_checkpoint_callback(checkpoint_dir, monitor='loss', save_weights_only=True, 
                            save_freq='epoch', verbose=False, mode='min'):
    if save_weights_only:
        cp_path = 'best_{}.h5'.format(monitor)
    else:
        cp_path = 'best_{}'.format(monitor)
    cb = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, cp_path),
                         monitor=monitor, save_best_only=True, mode=mode, verbose=verbose,
                         save_weights_only=save_weights_only, save_freq=save_freq)
    return cb
