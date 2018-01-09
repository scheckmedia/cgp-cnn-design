from keras.models import load_model, model_from_json
from trainer.cityscapes_trainer import CityscapesTrainer
from trainer.imagenet_trainer import ImageNetTrainer
from trainer.utils import fcn_to_fc
from keras.applications.mobilenet import DepthwiseConv2D
from fcn_utils.BilinearUpSampling import BilinearUpSampling2D
from fcn_utils.loss_function import softmax_sparse_crossentropy_ignoring_last_label
from layers.pad import PadZeros
from layers.shuffle import ChannelShuffle
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K
from fcn_utils.metrics import sparse_accuracy_ignoring_last_label
import json
import tensorflow as tf

def instantiate_class(cl, kwargs):
    d = cl.rfind(".")
    classname = cl[d+1:len(cl)]
    m = __import__(cl[0:d], globals(), locals(), [classname])
    return getattr(m, classname)(**kwargs)


if __name__ == '__main__':
    custom_objects = {
        'BilinearUpSampling2D': BilinearUpSampling2D,
        'PadZeros': PadZeros,
        'DepthwiseConv2D': DepthwiseConv2D,
        'ChannelShuffle': ChannelShuffle }

    input_shape = (None, 1024, 2048, 3)
    target_size = (1024, 2048)
    initial_epoch = 13

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        K.set_session(sess)

        model = load_model('tmp/cityscapes/model_child-1_score_0.450.hdf5', custom_objects=custom_objects)
        m = json.loads(model.to_json())
        m['config']['layers'][-1]['config']['target_size'] = list(target_size)
        m['config']['layers'][0]['config']['batch_input_shape'] = input_shape
        model = model_from_json(json.dumps(m), custom_objects=custom_objects)
        model.load_weights('tmp/cityscapes/trained_model.hdf5')
        model.summary()


        trainer = CityscapesTrainer(cs_root='/mnt/daten/Development/Cityscapes', verbose=1,
                                    input_shape=input_shape[1:4], target_size=target_size, batch_size=2,
                                    lr=[0.01], epochs=100)

        logger = CSVLogger('tmp/cityscapes/training.csv', append=initial_epoch > 0)
        rlp = ReduceLROnPlateau(monitor='loss')
        cp = ModelCheckpoint('tmp/cityscapes/trained_model.hdf5', save_best_only=True,  save_weights_only=True, verbose=1, monitor='loss')
        trainer(model, 0, callbacks=[logger, cp, rlp], initial_epoch=initial_epoch, every_n_epoch=10, skip_checks=True)
