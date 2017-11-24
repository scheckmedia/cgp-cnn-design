from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
import numpy as np
import operator
from trainer.trainer import ClassifyTrainer

class Cifar10Trainer(ClassifyTrainer):
    def __init__(self, batch_size=32, epochs=100, verbose=1):
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

    def __call__(self, model):
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
        optimizer = Adam(lr=0.001)
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
        steps = len(self.x_test) // self.batch_size
        history = model.fit_generator(generator=self.generator, steps_per_epoch=steps, epochs=self.epochs,
                                      validation_data=(self.x_test, self.y_test), workers=4, verbose=self.verbose)

        return np.max(history.history['val_acc'])
