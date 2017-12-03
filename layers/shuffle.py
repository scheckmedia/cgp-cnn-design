from keras import backend as K
from keras.engine.topology import Layer


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
                print(in_channels, self.groups_factor, in_channels % self.groups_factor)
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