from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, BatchNormalization, Add, Concatenate, Lambda, Activation
from keras.applications.mobilenet import DepthwiseConv2D
from cgp.cgp import CgpConfig, CGP
from evaluator.keras_evaluator import Evaluator
from trainer.cifar_trainer import Cifar10Trainer
from trainer.voc2012_trainer import Voc2012Trainer
from trainer.cityscapes_trainer import CityscapesTrainer
from layers.shuffle import ShuffleBlock
import math
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


# pylint: disable=line-too-long
function_mapping = {
    'pw_32': {'cls': Conv2D, 'args': {'filters': 32, 'kernel_size': 1, 'strides': 1, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'pw_64': {'cls': Conv2D, 'args': {'filters': 64, 'kernel_size': 1, 'strides': 1, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'pw_128': {'cls': Conv2D, 'args': {'filters': 128, 'kernel_size': 1, 'strides': 1, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'pw_256': {'cls': Conv2D, 'args': {'filters': 256, 'kernel_size': 1, 'strides': 1, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'pw_512': {'cls': Conv2D, 'args': {'filters': 512, 'kernel_size': 1, 'strides': 1, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'pw_1024': {'cls': Conv2D, 'args': {'filters': 1024, 'kernel_size': 1, 'strides': 1, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_32_stride_1': {'cls': Conv2D, 'args': {'filters': 32, 'kernel_size': 3, 'strides': 1, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_64_stride_1': {'cls': Conv2D, 'args': {'filters': 64, 'kernel_size': 3, 'strides': 1, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_32_stride_2': {'cls': Conv2D, 'args': {'filters': 32, 'kernel_size': 3, 'strides': 2, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_64_stride_2': {'cls': Conv2D, 'args': {'filters': 64, 'kernel_size': 3, 'strides': 2, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    #'conv_3x3_32_stride_3': {'cls': Conv2D, 'args': {'filters': 32, 'kernel_size': 3, 'strides': 3, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    #'conv_3x3_64_stride_3': {'cls': Conv2D, 'args': {'filters': 64, 'kernel_size': 3, 'strides': 3, 'kernel_initializer': 'he_uniform', 'padding': 'same'}, 'inputs': 1},
    'dw_3x3_stride_1': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 3, 'kernel_initializer': 'he_uniform', 'padding': 'same', 'strides': 1}, 'inputs': 1},
    'dw_5x5_stride_1': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 5, 'kernel_initializer': 'he_uniform', 'padding': 'same', 'strides': 1}, 'inputs': 1},
    'dw_7x7_stride_1': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 7, 'kernel_initializer': 'he_uniform', 'padding': 'same', 'strides': 1}, 'inputs': 1},
    'dw_3x3_stride_2': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 3, 'kernel_initializer': 'he_uniform', 'padding': 'same', 'strides': 2}, 'inputs': 1},
    'dw_5x5_stride_2': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 5, 'kernel_initializer': 'he_uniform', 'padding': 'same', 'strides': 2}, 'inputs': 1},
    'dw_7x7_stride_2': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 7, 'kernel_initializer': 'he_uniform', 'padding': 'same', 'strides': 2}, 'inputs': 1},
    'max_pooling_2x2': {'cls': MaxPool2D, 'args': {'pool_size': 2, 'padding': 'same'}, 'inputs': 1},
    #'max_pooling_3x3': {'cls': MaxPool2D, 'args': {'pool_size': 3, 'padding': 'same'}, 'inputs': 1},
    'avg_pooling_2x2': {'cls': AveragePooling2D, 'args': {'pool_size': 2, 'padding': 'same'}, 'inputs': 1},
    #'avg_pooling_3x3': {'cls': AveragePooling2D, 'args': {'pool_size': 3, 'padding': 'same'}, 'inputs': 1},

    'shuffle_block_g4_32': {'cls': ShuffleBlock, 'args': {'groups': 4, 'filters': 32}, 'inputs': 1},
    'shuffle_block_g4_64': {'cls': ShuffleBlock, 'args': {'groups': 4, 'filters': 64}, 'inputs': 1},
    'shuffle_block_g4_128': {'cls': ShuffleBlock, 'args': {'groups': 4, 'filters': 128}, 'inputs': 1},
    'shuffle_block_g4_256': {'cls': ShuffleBlock, 'args': {'groups': 4, 'filters': 256}, 'inputs': 1},
    'shuffle_block_g4_512': {'cls': ShuffleBlock, 'args': {'groups': 4, 'filters': 512}, 'inputs': 1},
    'shuffle_block_g4_1024': {'cls': ShuffleBlock, 'args': {'groups': 4, 'filters': 1024}, 'inputs': 1},

    'shuffle_block_g8_32': {'cls': ShuffleBlock, 'args': {'groups': 8, 'filters': 32}, 'inputs': 1},
    'shuffle_block_g8_64': {'cls': ShuffleBlock, 'args': {'groups': 8, 'filters': 64}, 'inputs': 1},
    'shuffle_block_g8_128': {'cls': ShuffleBlock, 'args': {'groups': 8, 'filters': 128}, 'inputs': 1},
    'shuffle_block_g8_256': {'cls': ShuffleBlock, 'args': {'groups': 8, 'filters': 256}, 'inputs': 1},
    'shuffle_block_g8_512': {'cls': ShuffleBlock, 'args': {'groups': 8, 'filters': 512}, 'inputs': 1},
    'shuffle_block_g8_1024': {'cls': ShuffleBlock, 'args': {'groups': 8, 'filters': 1024}, 'inputs': 1},

    'add': {'cls': Add, 'args': {}, 'inputs': 2},
    'concat': {'cls': Concatenate, 'args': {}, 'inputs': 2}
}

if __name__ == '__main__':
    # trainer = Cifar10Trainer(batch_size=256, epochs=60, verbose=1, lr=[0.001, 0.0005])
    #trainer = Voc2012Trainer(voc_root='/mnt/daten/Development/VOCdevkit/VOC2012/',
    #                         verbose=1, lr=[0.0005])
    trainer = CityscapesTrainer(cs_root='/mnt/daten/Development/Cityscapes_256', verbose=1, lr=[0.01])
    e = Evaluator(function_mapping, trainer, input_shape=trainer.input_shape, can_growth=True)
    functions, inputs = e.get_function_input_list()
    cfg = CgpConfig(rows=10, cols=30, level_back=10, functions=functions, function_inputs=inputs, mutation_rate=0.2)
    cgp = CGP(cfg, children=1)
    cgp.run(e, max_epochs=200000, save_best='tmp/parent.pkl')
