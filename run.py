from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Multiply, Add, Concatenate, SeparableConv2D
from keras.applications.mobilenet import DepthwiseConv2D
from cgp.cgp import CgpConfig, CGP
from evaluator.keras_evaluator import Evaluator
from trainer.cifar_trainer import Cifar10Trainer
from trainer.voc2012_trainer import Voc2012Trainer
from layers.shuffle import ChannelShuffle

import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# pylint: disable=line-too-long
function_mapping = {
    'pw_32': {'cls': Conv2D, 'args': {'filters': 32, 'kernel_size': 1, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'pw_64': {'cls': Conv2D, 'args': {'filters': 64, 'kernel_size': 1, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'pw_128': {'cls': Conv2D, 'args': {'filters': 128, 'kernel_size': 1, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'pw_256': {'cls': Conv2D, 'args': {'filters': 256, 'kernel_size': 1, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_32_stride_1': {'cls': Conv2D, 'args': {'filters': 32, 'kernel_size': 3, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_64_stride_1': {'cls': Conv2D, 'args': {'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_128_stride_1': {'cls': Conv2D, 'args': {'filters': 128, 'kernel_size': 3, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'dw_3x3_stride_1': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 3, 'padding': 'same', 'strides': 1}, 'inputs': 1},
    'dw_5x5_stride_1': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 5, 'padding': 'same', 'strides': 1}, 'inputs': 1},
    'dw_7x7_stride_1': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 7, 'padding': 'same', 'strides': 1}, 'inputs': 1},
    'max_pooling_2x2': {'cls': MaxPool2D, 'args': {'pool_size': 2}, 'inputs': 1},
    'avg_pooling_2x2': {'cls': AveragePooling2D, 'args': {'pool_size': 2}, 'inputs': 1},
    'channel_shuffle': {'cls': ChannelShuffle, 'args': {'groups_factor': 4}, 'inputs': 1},
    'add_2': {'cls': Add, 'args': {}, 'inputs': 2},
    'concat_2': {'cls': Concatenate, 'args': {}, 'inputs': 2},
    'add': {'cls': Add, 'args': {}, 'inputs': 2},
    'concat': {'cls': Concatenate, 'args': {}, 'inputs': 2}
}

if __name__ == '__main__':
    # trainer = Cifar10Trainer(batch_size=256, epochs=60, verbose=1, lr=[0.001, 0.0005])
    trainer = Voc2012Trainer(voc_root='/mnt/daten/Development/VOCdevkit/VOC2012/',
                             verbose=1, lr=[0.01, 0.005, 0.001, 0.0005])
    e = Evaluator(function_mapping, trainer, input_shape=trainer.input_shape, can_growth=True)
    functions, inputs = e.get_function_input_list()
    cfg = CgpConfig(rows=6, cols=30, level_back=10, functions=functions, function_inputs=inputs)
    cgp = CGP(cfg, children=1)
    cgp.run(e, max_epochs=200000, save_best='tmp/parent.pkl')