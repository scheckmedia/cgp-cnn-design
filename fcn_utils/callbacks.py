from keras.callbacks import Callback
from fcn_utils.evaluation import calculate_iou
import numpy as np

class MeanIoUCallback(Callback):
    def __init__(self, model, generator, steps, num_classes, every_n_epoch=None, on_end=True):
        self.model = model
        self.generator = generator
        self.steps = steps
        self.num_classes = num_classes
        self.every_n_epoch = every_n_epoch
        self.on_end = on_end
        self.mean_ious = []

    def __calculate(self, logs={}):
        conf_m, iou, mean_iou = calculate_iou(self.model, self.generator,  self.steps, self.num_classes)

        pixel_acc = np.sum(np.diag(conf_m)) / np.sum(conf_m)
        self.mean_ious += [mean_iou]
        logs['mean_iou'] = self.mean_ious[-1]

        print("mean IoU: %.4f --- pixel acc: %.4f" % (mean_iou, pixel_acc))
        return self.mean_ious[-1]

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 0 or self.every_n_epoch == -1 or (self.every_n_epoch is not None and epoch % self.every_n_epoch != 0):
            return

        return self.__calculate(logs)

    def on_train_end(self, logs=None):
        if self.on_end == False:
            return

        return self.__calculate(logs)