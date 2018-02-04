import tensorflow as tf
import keras.backend as K
from keras.applications.mobilenet import MobileNet
from trainer.utils import fcn_wrapper
from keras.utils import plot_model
import argparse
import sys, os
import yaml


def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

def mkdirs(loader, node):
    seq = loader.construct_sequence(node)
    dirs = ''.join([str(i) for i in seq])
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    return dirs

def get_class(cl, kwargs = None, instantiate = False):
    d = cl.rfind(".")
    classname = cl[d+1:len(cl)]
    m = __import__(cl[0:d], globals(), locals(), [classname])

    cls = getattr(m, classname)
    if isinstance(kwargs, dict) and instantiate:
        cls = cls(**kwargs)

    return cls


def parse_yaml(f):
    with open(f, 'r') as f:
        yaml.add_constructor('!join', join)
        yaml.add_constructor('!makedirs', mkdirs)

        config = yaml.load(f)

        # if 'custom_objects' in config:
        #     custom_objects = {}
        #     for o in config['custom_objects']:
        #         c = get_class(o['cls'])
        #         custom_objects[o['name']] = c

        #config['custom_objects'] = custom_objects

        return config


if __name__ == '__main__':
    print("args %s" % str(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="training configuration file")
    args = parser.parse_args()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        K.set_session(sess)

        config = parse_yaml(args.config)

        if 'run' not in config:
            raise ValueError('you need to setup a run section in your yaml')

        trainer = config['run']['trainer']
        kwds = config['run']['kwds']
        trainer(**kwds)

        # m = MobileNet(input_shape=(224, 224, 3), include_top=False, weights=None)
        # wrapper = fcn_wrapper(m, skip_connections=['conv_dw_1_relu', 'conv_pw_6_relu'])
        # plot_model(wrapper, to_file='/Users/tobi/Desktop/model.png', show_shapes=True)

        #wrapper = fcn_wrapper('/Volumes/homes/jupyter/Masterthesis/cgp_ann_design/tmp/cityscapes/model_child-1_score_0.482.hdf5',
        #                      skip_connections=['pw_64_id_15_act', 'dw_5x5_stride_2_id_75_bn'])
        #plot_model(wrapper, to_file='/Users/tobi/Desktop/model.png', show_shapes=True)

        #trainer = CityscapesTrainer(cs_root='/mnt/daten/Development/Cityscapes', verbose=1,
        #                            input_shape=input_shape[1:4], target_size=target_size, batch_size=2,
        #                            lr=[0.01], epochs=100)

        #logger = CSVLogger('tmp/cityscapes/training.csv', append=initial_epoch > 0)
        #rlp = ReduceLROnPlateau(monitor='loss')
        #cp = ModelCheckpoint('tmp/cityscapes/trained_model.hdf5', save_best_only=True,  save_weights_only=True, verbose=1, monitor='loss')
        #trainer(model, 0, callbacks=[logger, cp, rlp], initial_epoch=initial_epoch, every_n_epoch=10, skip_checks=True)
