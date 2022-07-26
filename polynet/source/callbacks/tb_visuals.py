import io
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from source.models.gauss_layer import GaussFilter


def plot_cnn_layer(kernel, bias=None, title=None, kernel_ylim=None, bias_ylim=None):
    fig, ax = plt.subplots(1, 2 if bias is not None else 1, figsize=(10, 10/1.6))
    ax = ax if bias is not None else [ax]
    sns.lineplot(ax=ax[0], x=range(len(kernel)), y=kernel, lw=2)
    ax[0].set_xlabel('Kernel index', fontsize=18)
    ax[0].set_title('Kernel', fontsize=18)
    if kernel_ylim:
        ax[0].set_ylim(kernel_ylim)
    if bias is not None:
        sns.lineplot(ax=ax[1], x=range(len(bias)), y=bias, lw=2)
        ax[1].set_xlabel('Output index', fontsize=18)
        ax[1].set_title('Bias'.format(title), fontsize=18)
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        if bias_ylim:
            ax[1].set_ylim(bias_ylim)
    if title:
        fig.suptitle('{}'.format(title), fontsize=20)
    plt.tight_layout()
    return fig


def plot_mlp_layer(kernel, bias=None, title=None, kernel_clim=None,
                   bias_ylim=None):
    fig, ax = plt.subplots(1, 2 if bias is not None else 1, figsize=(10, 10 / 1.6))
    ax = ax if bias is not None else [ax]
    if kernel_clim is None:
        kernel_clim = (None, None)
    sns.heatmap(ax=ax[0], data=kernel, vmin=kernel_clim[0], vmax=kernel_clim[1],
                cbar_kws={'orientation': 'vertical'})
    ax[0].set_xlabel('Input index', fontsize=18)
    ax[0].set_ylabel('Output index', fontsize=18)
    ax[0].set_title('Kernel', fontsize=18)
    if bias is not None:
        sns.lineplot(ax=ax[1], x=range(len(bias)), y=bias, lw=2)
        ax[1].set_xlabel('Output index', fontsize=18)
        ax[1].set_title('Bias', fontsize=18)
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        if bias_ylim:
            ax[1].set_ylim(bias_ylim)
    if title:
        fig.suptitle('{}'.format(title), fontsize=20)
    plt.tight_layout()
    return fig


def fig_to_tbpng(figure):
    with io.BytesIO() as buf:
        figure.savefig(buf, format='png')
        buf.seek(0)
        img = tf.image.decode_png(buf.getvalue(), channels=4)
    plt.close(figure)
    return tf.expand_dims(img, 0)


class TBPlotter(keras.callbacks.Callback):
    def __init__(self, log_dir, on_epoch_begin=True):
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.exec_time = 'begin' if on_epoch_begin else 'end'
        super(TBPlotter, self).__init__()

    def gen_tb_images(self, epoch, logs=None):
        with self.file_writer.as_default(step=epoch):
            for layer in self.model.layers:
                if isinstance(layer, GaussFilter):
                    layer_ = self.model.get_layer(layer.name)
                    fig = plot_cnn_layer(
                        layer_.get_config()['kernel'].numpy().squeeze(),
                        kernel_ylim=[-1.05, 1.05],
                        title=layer.name)
                    img = fig_to_tbpng(fig)
                    tf.summary.image(layer.name, img)
                if isinstance(layer, keras.layers.Dense):
                    layer_ = self.model.get_layer(layer.name)
                    kernel_t, bias = layer_.kernel.numpy().T, layer_.bias
                    if bias is not None:
                        bias = bias.numpy().squeeze()
                    fig = plot_mlp_layer(kernel_t, bias,
                                         kernel_clim=[-1, 1], bias_ylim=[-1, 1],
                                         title=layer.name)
                    img = fig_to_tbpng(fig)
                    tf.summary.image(layer.name, img)

    def on_epoch_begin(self, epoch, logs=None):
        if self.exec_time == 'begin':
            self.gen_tb_images(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.exec_time == 'end':
            self.gen_tb_images(epoch, logs=logs)
