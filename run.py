from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Multiply, Add, Concatenate, SeparableConv2D
from keras.applications.mobilenet import DepthwiseConv2D
from cgp.cgp import CgpConfig, CGP
from evaluator.keras_evaluator import Evaluator
from trainer.cifar_trainer import Cifar10Trainer


function_mapping = {
    'pw_32': {'cls': Conv2D, 'args': {'filters': 32, 'kernel_size': 1, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'pw_64': {'cls': Conv2D, 'args': {'filters': 64, 'kernel_size': 1, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'pw_128': {'cls': Conv2D, 'args': {'filters': 128, 'kernel_size': 1, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_32_stride_1': {'cls': Conv2D, 'args': {'filters': 32, 'kernel_size': 3, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_64_stride_1': {'cls': Conv2D, 'args': {'filters': 64, 'kernel_size': 3, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'conv_3x3_128_stride_1': {'cls': Conv2D, 'args': {'filters': 128, 'kernel_size': 3, 'strides': 1, 'padding': 'same'}, 'inputs': 1},
    'dw_3x3_stride_1': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 3, 'padding': 'same', 'strides': 1}, 'inputs': 1},
    'dw_5x5_stride_1': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 5, 'padding': 'same', 'strides': 1}, 'inputs': 1},
    'dw_7x7_stride_1': {'cls': DepthwiseConv2D, 'args': {'kernel_size': 7, 'padding': 'same', 'strides': 1}, 'inputs': 1},
    'max_pooling_2x2': {'cls': MaxPool2D, 'args': {'pool_size': 2}, 'inputs': 1},
    'avg_pooling_2x2': {'cls': AveragePooling2D, 'args': {'pool_size': 2}, 'inputs': 1},
    # 'max_pooling_2x2_2': {'cls': MaxPool2D, 'args': {'pool_size': 2}, 'inputs': 1},
    # 'avg_pooling_2x2_2': {'cls': AveragePooling2D, 'args': {'pool_size': 2}, 'inputs': 1},
    'add': {'cls': Add, 'args': {}, 'inputs': 2},
    'concat': {'cls': Concatenate, 'args': {}, 'inputs': 2}
}

if __name__ == '__main__':
    trainer = Cifar10Trainer(batch_size=128, epochs=60, verbose=1, lr=[0.1, 0.05, 0.01])
    e = Evaluator(function_mapping, trainer, input_shape=trainer.input_shape)
    functions, inputs = e.get_function_input_list()
    cfg = CgpConfig(rows=5, cols=30, level_back=10, functions=functions, function_inputs=inputs)
    cgp = CGP(cfg, children=1)
    cgp.run(e, max_epochs=200000, save_best='tmp/parent.pkl')