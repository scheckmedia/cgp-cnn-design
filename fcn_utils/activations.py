from keras import backend as K
from keras.activations import get as activations_get


def softmax_4d(x):
    axis = -1 if K.image_dim_ordering() == "tf" else 1
    e = K.exp(x - K.max(x, axis=axis, keepdims=True))
    s = K.sum(e, axis=axis, keepdims=True)
    return e / s


def get(identifier):
    if identifier == "softmax_4_dimension":
        return softmax_4d
    return activations_get(identifier)