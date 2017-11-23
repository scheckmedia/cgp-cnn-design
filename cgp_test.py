import numpy as np
from cgp.utils import individual_to_keras_model
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import Conv2D, Maximum, MaxPool2D, AveragePooling2D, Add, Concatenate
import time

from cgp.cgp import CgpConfig, CGP

def evaluator(child, child_number):
    individual_to_keras_model(child)
    for i in range(0, np.random.randint(2, 5)):
        print("call and simulate some heavy work at child %d!" % child_number)
        time.sleep(1)
    return np.random.randint(0, 10)

functions = [
    Conv2D(32, kernel_size=3, strides=2, name='conv_3x3_32_stride_2', padding='same'),
    Conv2D(64, kernel_size=3, strides=2, name='conv_3x3_64_stride_2', padding='same'),
    Conv2D(128, kernel_size=3, strides=2, name='conv_3x3_128_stride_2', padding='same'),
    MaxPool2D(pool_size=2, name='max_pooling_2x2'),
    AveragePooling2D(pool_size=2, name='avg_pooling_2x2'),
    Add(name='add'),
    Maximum(name='maximum'),
    Concatenate(name='concat')
]
inputs = [1, 1, 1, 1, 1, 2, 2, 2]
cfg = CgpConfig(rows=5, cols=30, level_back=10, functions=functions, function_inputs=inputs)
cfg.num_input = 1
cfg.num_output = 1
cgp = CGP(cfg)
cgp.run(evaluator, max_epochs=20)