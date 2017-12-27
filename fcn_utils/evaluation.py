import numpy as np
from pylab import *
from PIL import Image
from multiprocessing import Process, Queue


def __confusion_matrix(q, label, preds, num_classes, ignore=255):
    conf_m = zeros((num_classes, num_classes), dtype=float)
    for label, pred in zip(label, preds):
        # pred = np.argmax(pred, axis=-1).astype(np.uint8)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        for p, l in zip(flat_pred, flat_label):
            if l == ignore:
                continue
            if l < num_classes and p < num_classes:
                conf_m[l, p] += 1

    q.put(conf_m)


# based on https://github.com/aurora95/Keras-FCN/blob/master/evaluate.py
def calculate_iou(model, generator, steps, num_classes, num_workers=6):
    print('\ncalculating mean IoU')
    generator.reset()

    preds = []
    labels = []
    for i in range(steps):
        x, y_true = generator.next()
        y_pred = model.predict(x)
        y_pred = np.argmax(y_pred, axis=-1).astype(np.uint8)
        preds += list(y_pred)
        labels += list(y_true)

    q = Queue()
    workers = []
    num_items_per_worker = len(preds) // num_workers
    print('calculate confusion matrix for %d predictions on %d CPUs with %d items per worker' %
          (len(preds), num_workers, num_items_per_worker))
    for i in range(num_workers):
        start = i * num_items_per_worker
        end = i * num_items_per_worker + num_items_per_worker

        if i == num_workers - 1:
            end += num_items_per_worker % num_workers

        pred = preds[start:end]
        label = labels[start:end]
        ignore = generator.label_cval
        workers.append(Process(target=__confusion_matrix,
                               args=(q, label, pred, num_classes, ignore)))

    for t in workers:
        t.start()

    for t in workers:
        t.join()

    conf_matrix = None

    while not q.empty():
        m = q.get()

        if conf_matrix is None:
            conf_matrix = m
        else:
            conf_matrix = np.add(m, conf_matrix)

    intersection = np.diag(conf_matrix)
    union = np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - intersection
    iou = intersection/union
    mean_iou = np.mean(iou)
    print(iou)
    return conf_matrix, iou, mean_iou
