import tensorflow as tf
import keras.backend as K
from keras.models import clone_model
from keras.datasets import cifar10
from keras.utils import to_categorical, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import Dense, Flatten
import numpy as np
import operator
from trainer.trainer import ClassifyTrainer
import os
import csv


class Cifar10Trainer(ClassifyTrainer):
    def __init__(self, batch_size=32, epochs=100, verbose=0, lr=None, model_path='tmp/', stats_path='tmp/'):
        """
        A trainer class for the Cifar 10 dataset

        Parameters
        ----------
        batch_size: int(32)
            batch_size for cifar10 dataset
        epochs: int(100)
            number of epochs are used for each training process
        verbose: int(1)
            see keras.model.fit_generator
        """

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        input_shape = x_train.shape[1:]

        ClassifyTrainer.__init__(self, batch_size=batch_size, num_classes=10, input_shape=input_shape,
                                 epochs=epochs, verbose=verbose)

        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)

        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

        datagen.fit(self.x_train)
        self.generator = datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size)
        self.model_path = model_path
        self.stats_path = stats_path
        self._csv_file = os.path.join(self.stats_path, 'stats.csv')
        self.learning_rates = lr

        with open(self._csv_file, 'w') as f:
            header = ['epoch', 'val_acc', 'val_loss', 'params', 'flops', 'score']
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(header)

    def comp(self, parent, child):
        """
        compares a parent and a child and decides whether a child is beter than a parent
        Parameters
        ----------
        parent: float
            parent score
        child: float
            child score

        Returns
        -------
            True if a child is better than a parent otherwise False
        """
        return operator.lt(parent, child)

    def append_output_layer(self, output):
        """
        appends the last layer (dense, softmax) to an output layer
        Parameters
        ----------
        output: keras.layers.layer
            layer where additional layer should be connected

        Returns
        -------
        keras.layers.layer

        """
        x = Flatten()(output)
        x = Dense(self.num_classes, activation='softmax', name='softmax')(x)
        return x

    def model_improved(self, model, score):
        """

        Parameters
        ----------
        model: keras.models.Model
            model which has the best score for a CGP iteration

        Returns
        -------

        """
        base_path = os.path.abspath(self.model_path)
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        f = os.path.join(self.model_path, 'model_%s_score_%.3f.hdf5' % (model.name, score))
        model.save(f)
        plot_model(model, show_shapes=True,
                   to_file=os.path.join(self.model_path, 'model_%s_score_%.3f.png' % (model.name, score)))
        print("save model %s with score %.5f to file" % (f, score))

    def __call__(self, model, epoch):
        """
        starts the training of a keras model

        Parameters
        ----------
        model: keras.models.Model
            a keras model which will be trained

        Returns
        -------
        float
            score of the best model

        """
        run_meta = tf.RunMetadata()

        callbacks = []
        if self.learning_rates:
            lr_idx = (self.epochs // len(self.learning_rates))
            lr_scheduler = LearningRateScheduler(lambda epoch: self.learning_rates[epoch // lr_idx])
            callbacks.append(lr_scheduler)

        callbacks.append(EarlyStopping(monitor='val_acc', mode='max', patience=10, min_delta=0.001, verbose=1))

        _ = clone_model(model, input_tensors=tf.placeholder('float32', shape=(1,32,32,3)))
        option_builder = tf.profiler.ProfileOptionBuilder
        profiler = tf.profiler.profile

        opt = option_builder.float_operation()
        opt['output'] = 'none'
        flops = profiler(K.get_session().graph, run_meta=run_meta, options=opt)

        opt = option_builder.trainable_variables_parameter()
        opt['output'] = 'none'
        params = profiler(K.get_session().graph, run_meta=run_meta, options=opt)

        # it seems that maac in tensorflow is counted as two operations
        # I divide the flops by two to get a nearly similar value
        total_flops, total_params = flops.total_float_ops // 2, params.total_parameters
        max_params = 4 * 10**6  # max number of params  # e.g. 3.3M of the MobileNet or 25.56M of ResNet 50
        max_flops = 10 * 10**6    # max number of flops # e.g. 3858M of ResNet 50

        params_factor = 1 - (min(max_params, total_params) / max_params)
        flops_factor = 1 - (min(max_flops, total_flops) / max_flops)

        if params_factor <= 0.0 or flops_factor <= 0.0:
            return 0

        optimizer = SGD(lr=0.01, decay=5e-4, momentum=0.9)
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)

        steps = len(self.x_test) // self.batch_size
        history = model.fit_generator(generator=self.generator, steps_per_epoch=steps, epochs=self.epochs,
                                      validation_data=(self.x_test, self.y_test), workers=4, verbose=self.verbose,
                                      callbacks=callbacks)

        acc = np.max(history.history['val_acc'])
        loss = np.min(history.history['val_loss'])

        score = params_factor * flops_factor * acc

        print("\n%s" % ("-" * 100))
        print("acc: %.2f ---> params: %d, %.2f ---> flops: %s, %.2f ---> score: %.2f" %
              (acc, total_params, params_factor, "{:,}".format(total_flops), flops_factor, score))
        print("%s\n" % ("-" * 100))

        if self.stats_path:
            if not os.path.exists(self.stats_path):
                os.mkdir(self.stats_path)

            with open(self._csv_file, 'a') as f:
                header = [epoch, acc, loss, total_params, total_flops, score]
                w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                w.writerow(header)

        return score
