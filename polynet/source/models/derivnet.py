import tensorflow as tf
import keras
from source.models.base_model import ModelFactory, BaseModel
from source.models.gauss_layer import GaussFilter
from source.models.mvn import MVN
from source.models.pad_layer import Pad1D


class DerivNet(ModelFactory):
    def __init__(self, input_dim=(50, 1), deriv_orders=None, kernel_width=16, stride=1, eps=1e-3,
                 pool_size=4, pool_stride=4, mlp_dims=None, output_dim=2, dropout_rate=0.01,
                 mean_norm=True, var_norm=True, **kwargs):
        if deriv_orders is None:
            deriv_orders = [1]
        if mlp_dims is None:
            mlp_dims = [25, 10, 10]
        self.input_dim = input_dim
        self.deriv_orders = deriv_orders
        self.kernel_width = kernel_width
        self.stride = stride
        self.eps = eps
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.mlp_dims = mlp_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.mean_norm = mean_norm
        self.var_norm = var_norm
        super().__init__()

    def create_model(self):
        inputs = keras.layers.Input(shape=self.input_dim, name='input')

        # MVN Layer if applicable
        if self.var_norm or self.mean_norm:
            mvn = MVN(mean_norm=self.mean_norm,
                      var_norm=self.var_norm,
                      update=True, name='mvn')(inputs)
        else:
            mvn = inputs

        # Pad inputs to maintain same input dim after convolutions
        padded_inputs = Pad1D(padding=(self.kernel_width // 2, self.kernel_width // 2 - 1),
                              name='pad')(mvn)

        # Pseudo-Derivatives
        derivs = []
        for order in self.deriv_orders:
            gf = GaussFilter(kernel_width=self.kernel_width,
                             order=order,
                             stride=self.stride,
                             eps=self.eps, name='deriv_{}'.format(order))(padded_inputs)

            maxp = tf.keras.layers.MaxPool1D(pool_size=self.pool_size,
                                             strides=self.pool_stride,
                                             name='maxpool_{}'.format(order))(gf)
            derivs.append(maxp)

        # Concatenate all AG layers into one resulting tensor, pass through MLP
        concat = tf.keras.layers.Concatenate(axis=1, name='concat')(derivs)
        mlp = tf.keras.layers.Reshape(concat.shape[1:-1], name='reshape')(concat)

        # MLP portion of model
        for layer_idx, layer_dim in enumerate(self.mlp_dims):
            mlp = tf.keras.layers.Dense(layer_dim,
                                        activation=None,
                                        name='dense{}'.format(layer_idx))(mlp)
            if self.dropout_rate > 0:
                mlp = tf.keras.layers.Dropout(rate=self.dropout_rate,
                                              name='dropout{}'.format(layer_idx))(mlp)
            mlp = tf.keras.layers.PReLU(shared_axes=False,
                                        name='act_fn{}'.format(layer_idx))(mlp)

        # Output (logit values of class membership)
        output = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='output')(mlp)

        # Create and return model
        return BaseModel(inputs=inputs, outputs=output, name='DerivNet')
