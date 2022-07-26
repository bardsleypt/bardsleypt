import h5py
from abc import ABC, abstractmethod
from keras.engine.training import Model
from keras.saving.hdf5_format import save_attributes_to_hdf5_group, _legacy_weights
from keras import backend, __version__ as keras_version


class ModelFactory(ABC):
    @abstractmethod
    def create_model(self, *args, **kwargs):
        pass


class BaseModel(Model, ABC):
    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Override the save_weights method ONLY if the filepath ends in .h5. The
        # following code is a verbatim copy of the super-class method for these files,
        # other than track_times=False in the create_dataset calls.
        if filepath.endswith('.h5'):
            layers = self.layers
            with h5py.File(filepath, 'w') as f:
                save_attributes_to_hdf5_group(
                    f, 'layer_names', [layer.name.encode('utf8') for layer in layers])
                f.attrs['backend'] = backend.backend().encode('utf8')
                f.attrs['keras_version'] = str(keras_version).encode('utf8')

                # Sort model layers by layer name to ensure that group names are strictly
                # growing to avoid prefix issues.
                for layer in sorted(layers, key=lambda x: x.name):
                    g = f.create_group(layer.name)
                    weights = _legacy_weights(layer)
                    weight_values = backend.batch_get_value(weights)
                    weight_names = [w.name.encode('utf8') for w in weights]
                    save_attributes_to_hdf5_group(g, 'weight_names', weight_names)
                    for name, val in zip(weight_names, weight_values):
                        param_dset = g.create_dataset(name, val.shape, dtype=val.dtype,
                                                      track_times=False)
                        if not val.shape:
                            # scalar
                            param_dset[()] = val
                        else:
                            param_dset[:] = val
        else:
            super().save_weights(filepath, overwrite=overwrite,
                                 save_format=save_format, options=options)
