from keras.models import Model
from keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D, Activation, Dense

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