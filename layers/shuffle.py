from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv2D, Concatenate, Lambda, Activation, BatchNormalization
from keras.applications.mobilenet import DepthwiseConv2D


class ChannelShuffle(Layer):
    def __init__(self, groups=None, groups_factor=8, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self.groups = groups
        self.groups_factor = groups_factor
        self.trainable = False

    def build(self, input_shape):
        super(ChannelShuffle, self).build(input_shape)

    def call(self, x):
        height, width, in_channels = x.shape.as_list()[1:]

        if self.groups is None:
            if in_channels % self.groups_factor:
                raise ValueError("%s %% %s" % (in_channels, self.groups_factor))

            self.groups = in_channels // self.groups_factor

        channels_per_group = in_channels // self.groups

        x = K.reshape(x, [-1, height, width, self.groups, channels_per_group])
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
        x = K.reshape(x, [-1, height, width, in_channels])

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'groups': self.groups, 'groups_factor': self.groups_factor}
        base_config = super(ChannelShuffle, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SliceLayer(Layer):
    def __init__(self, start=0, items=8, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.start = start
        self.items = items
        self.trainable = False

    def build(self, input_shape):
        super(SliceLayer, self).build(input_shape)

    def call(self, x):
        return x[:, :, :, self.start: self.start + self.items]

    def compute_output_shape(self, input_shape):
        height, width, in_channels = input_shape[1:]
        return input_shape[0], height, width, self.items

    def get_config(self):
        config = {'start': self.start, 'items': self.items}
        base_config = super(SliceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class ShuffleBlock:
    def __init__(self, groups=1,  filters=8, name=""):
        self.groups = groups
        self.out_channels = filters
        self.name = name

    def build(self, input_shape):
        super(ShuffleBlock, self).build(input_shape)

    def __call__(self, x):
        height, width, in_channels = x.shape.as_list()[1:]
        if not in_channels % self.groups == 0:
            self.groups = 1

        # bottleneck_channels = in_channels // self.groups
        x = _group_conv(x, in_channels, out_channels=self.out_channels,
                        groups=self.groups,
                        name='%s/g%d' % (self.name, self.groups))
        x = BatchNormalization(axis=-1, name='%s/bn_gconv_1' % self.name)(x)
        x = Activation('relu', name='%s/relu_gconv_1' % self.name)(x)
        x = ChannelShuffle(groups=self.groups, name='%s/channel_shuffle' % self.name)(x)

        return x

    def compute_output_shape(self, input_shape):
        height, width, in_channels = input_shape[1:]
        return input_shape[0], height, width, self.out_channels


def _group_conv(x, in_channels, out_channels, groups, kernel=1, stride=1, name=''):
    """
    grouped convolution
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    in_channels:
        number of input channels
    out_channels:
        number of output channels
    groups:
        number of groups per channel
    kernel: int(1)
        An integer or tuple/list of 2 integers, specifying the
        width and height of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    stride: int(1)
        An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
        Can be a single integer to specify the same value for all spatial dimensions.
    name: str
        A string to specifies the layer name
    Returns
    -------
    """
    if groups == 1:
        return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
                      use_bias=False, strides=stride, name=name)(x)

    # number of intput channels per group
    ig = in_channels // groups
    group_list = []

    assert out_channels % groups == 0
    
    for i in range(groups):
        offset = i * ig
        group = SliceLayer(start=offset, items=ig, name='%s/g%d_slice' % (name, i))(x)
        # group = Lambda(slice_tensor, arguments={'start': offset, 'num_items': ig}, name='%s/g%d_slice' % (name, i))(x)
        group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride,
                                 use_bias=False, padding='same', name='%s/g%d' % (name, i))(group))
    return Concatenate(name='%s/concat' % name)(group_list)