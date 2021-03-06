from keras.callbacks import Callback
from fcn_utils.evaluation import calculate_iou
import numpy as np
import os

class MeanIoUCallback(Callback):
    def __init__(self, model, generator, steps, num_classes, every_n_epoch=None, on_end=True,
                 save_path=None, early_stop=None):
        self.model = model
        self.generator = generator
        self.steps = steps
        self.num_classes = num_classes
        self.every_n_epoch = every_n_epoch
        self.on_end = on_end
        self.mean_ious = []
        self.save_path = save_path
        self.early_stop = early_stop

    def __calculate(self, logs={}):
        conf_matrix, iou, mean_iou = calculate_iou(self.model, self.generator,  self.steps, self.num_classes)

        pixel_acc = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        self.mean_ious += [mean_iou]
        logs['mean_iou'] = self.mean_ious[-1]

        print("mean IoU: %.4f --- pixel acc: %.4f" % (mean_iou, pixel_acc))

        if self.save_path:
            np.save(os.path.join(self.save_path, 'conf_matrix'), conf_matrix)

        return self.mean_ious[-1]

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 0 or self.every_n_epoch == -1 or (self.every_n_epoch is not None and epoch % self.every_n_epoch != 0):
            return

        miou = self.__calculate(logs)

        if self.early_stop is not None and epoch in self.early_stop:
            if self.early_stop[epoch] > miou:
                self.on_end = False
                self.model.stop_training = True
                print("early stop because of bad mIoU")

        return miou

    def on_train_end(self, logs=None):
        if not self.on_end:
            return

        return self.__calculate(logs)