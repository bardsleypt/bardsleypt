import keras
from source.models.base_model import ModelFactory, BaseModel
from source.models.mvn import MVN


class MLPNet(ModelFactory):
    def __init__(self, input_dim=50, hidden_dims=None, output_dim=2,
                 dropout_rate=0.02, mean_norm=True, var_norm=True, **kwargs):
        if hidden_dims is None:
            hidden_dims = [25, 10, 10]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
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
        
        # Loop to create interior MLP layers
        mlp = mvn
        for layer_idx, layer_dim in enumerate(self.hidden_dims):
            mlp = keras.layers.Dense(layer_dim, activation=None,
                                     name='layer{}'.format(layer_idx))(mlp)
            if self.dropout_rate > 0:
                mlp = keras.layers.Dropout(rate=self.dropout_rate,
                                           name='dropout{}'.format(layer_idx))(mlp)
            mlp = keras.layers.PReLU(shared_axes=[1],
                                     name='act_fn{}'.format(layer_idx))(mlp)

        # Output (logit values of class membership)
        outputs = keras.layers.Dense(self.output_dim, name='output')(mlp)

        # Create and return model
        return BaseModel(inputs=inputs, outputs=outputs, name='MLPNet')
