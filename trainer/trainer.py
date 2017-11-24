import operator


class ClassifyTrainer:
    def __init__(self, batch_size=32, num_classes=100, input_shape=(32,32,3), epochs=100, verbose=1):
        """
        A trainer class will be called inside a evaluation process and contains the whole logic
        to train a model

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
        raise NotImplementedError("You need to implement __call__ in your Trainer class")