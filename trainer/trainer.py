import operator


class ClassifyTrainer:
    """
    A trainer class will be called inside a evaluation process and contains the whole logic
    to train a model
    """

    def __init__(self, batch_size=32, num_classes=100, input_shape=(32,32,3), epochs=100, verbose=1):
        """
        Parameters
        ----------
        batch_size: int(32)
            batch_size for cifar10 dataset
        num_classes: int(100)
            classes in a dataset
        input_shape: tuple(32,32,3)
            shape of an input
        epochs: int(100)
            number of epochs are used for each training process
        verbose: int(1)
            see keras.model.fit_generator
        """
        self.verbose = verbose
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.worst = float('-inf')

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
        return output

    def model_improved(self, model, score):
        pass

    def __call__(self, model, epoch, initial_epoch=0, callbacks=None, skip_checks=False):
        """
        starts the training of a keras model

        Parameters
        ----------
        model: keras.models.Model
            a keras model which will be trained
        epoch: int
            cgp epoch - just for the csv logging not necessary for plain training
        callbacks: list(keras.callbacks)
            a list of keras callbacks
        skip_checks:
            deactivates some checks which are required in cgp search mode e.g. if the flops are to high

        Returns
        -------
        float
            score of the best model

        """
        raise NotImplementedError("You need to implement __call__ in your Trainer class")