import sys, os
sys.path.insert(0, '..')

from keras.models import Model, load_model, model_from_json
from keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D, Activation, Dense, Conv2D, MaxPooling2D, add
from keras.applications.mobilenet import DepthwiseConv2D, relu6
from layers.pad import PadZeros
from layers.shuffle import ShuffleBlock, SliceLayer, ChannelShuffle
from fcn_utils.BilinearUpSampling import BilinearUpSampling2D
import keras.backend as K
import json


def fcn_to_fc(model, dense=False, pooling='max'):
    num_classes = model.layers[-2].input.shape.as_list()[-1]
    x = model.layers[-2].output

    if pooling == 'max':
        x = GlobalMaxPool2D()(x)
    else:
        x = GlobalAveragePooling2D()(x)

    if dense:
        x = Dense(num_classes, activation='softmax', name='softmax')(x)
    else:
        x = Activation('softmax', name='softmax')(x)

    return Model(inputs=model.input, outputs=x)


def fcn_wrapper(model, is_fcn=False, input_shape=(1024, 2048, 3), target_size=(1024, 2048),
                skip_connections=None, load_weights=None, num_classes=19, custom_objects=None):
    default_custom_objects = {
        'BilinearUpSampling2D': BilinearUpSampling2D,
        'PadZeros': PadZeros,
        'DepthwiseConv2D': DepthwiseConv2D,
        'relu6': relu6,
        'SliceLayer': SliceLayer,
        'ChannelShuffle': ChannelShuffle,
        'ShuffleBlock': ShuffleBlock}

    if not isinstance(custom_objects, dict):
        custom_objects = {}

    custom_objects = {**custom_objects, **default_custom_objects}

    if isinstance(model, str):
        model = load_model(model, custom_objects=custom_objects)

    if not isinstance(model, Model):
        raise TypeError('model is invalid')





    # workaround to change input and ouput size of an model
    m = json.loads(model.to_json())

    if is_fcn:
        m['config']['layers'][-1]['config']['target_size'] = list(target_size)

    m['config']['layers'][0]['config']['batch_input_shape'] = [None] + list(input_shape)
    model = model_from_json(json.dumps(m), custom_objects=custom_objects)

    pool = []
    if is_fcn:
        model.layers.pop()
        model.layers.pop()

    # get feature_extractor output
    output_shape = model.layers[-1].output_shape[-3:-1]
    upsample_total = tuple(map(lambda x, y: int(x / y), target_size, output_shape))
    output_conv = model.layers[-1].output
    output_up = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='relu', name="conv_last")(output_conv)

    # upscaling if output size should be greater than feature output size
    if 0 not in upsample_total:
        output_up = BilinearUpSampling2D(size=upsample_total, name="upsample_total")(output_up)
    else:
        # downscaling if output size should be less than feature output size
        pool_size = tuple(
            map(lambda x, y: int(y / x),
                output_shape,
                target_size)
        )

        output_up = MaxPooling2D(pool_size=pool_size, name='downsample_total')(output_up)

    pool.append(output_up)

    if skip_connections:
        for idx, s in enumerate(skip_connections):
            try:
                current_layer = model.get_layer(s).output
            except ValueError:
                raise ValueError('invalid skip connection. layer "%s" does not exist in model' % s)

            previous_layer = pool[-1]
            scale_factor = tuple(
                map(lambda x, y: int(y / x),
                    K.int_shape(current_layer)[-3:-1],
                    K.int_shape(previous_layer)[-3:-1]
                )
            )

            if 0 in scale_factor:
                # we need downsampling
                pool_size = tuple(
                    map(lambda x, y: int(x / y),
                        K.int_shape(current_layer)[-3:-1],
                        K.int_shape(previous_layer)[-3:-1]
                    )
                )
                scale_factor = (1, 1)

                layer_name = '%s_skip_downsample_%d' % (current_layer.name.split('/')[0], idx)
                current_layer = MaxPooling2D(pool_size=pool_size,
                                             name=layer_name)(current_layer)

            layer_name = '%s_skip_conv_%d' % (current_layer.name.split('/')[0], idx)
            x = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='relu', name=layer_name)(current_layer)

            if scale_factor != (1, 1):
                layer_name = "skip_%s_%d" % (current_layer.name.split('/')[0], idx)
                x = BilinearUpSampling2D(size=scale_factor, name=layer_name)(x)

            merge = add([x, previous_layer], name="add_%s_with_%s" % (x.name.split('/')[0], pool[-1].name.split('/')[0]))
            pool.append(merge)

    #x = Conv2D(num_classes, kernel_size=(1, 1), activation='relu', padding='same')()
    #x = BilinearUpSampling2D(target_size=target_size)(pool[-1])
    model = Model(inputs=model.input, outputs=pool[-1])

    if load_weights:
        model.load_weights(load_weights, by_name=True)

    model.summary()
    return model