import yaml, os

_lables = None

class Label:
    def __init__(self, *args):
        if len(args) != 8:
            raise ValueError('invalid arguments')

        self.name = args[0]
        self.id = args[1]
        self.trainId = args[2]
        self.category = args[3]
        self.catId = args[4]
        self.hasInstances = args[5]
        self.ignoreInEval = args[6]
        self.color = eval(args[7])

    @classmethod
    def from_list(cls, labels):
        instances = []
        for l in labels:
            instances += [Label(*l)]

        return instances


def _get_labels():
    global _lables

    if _lables == None:
        p = os.path.join(os.path.dirname(__file__), 'cityscapes_labels.yml')
        with open(p, 'r') as f:
            labels = yaml.load(f)
            return Label.from_list(labels['labels'])


def label_mapping():
    return {label.id: label.trainId for label in _get_labels()}


def label_names(attribute='trainId'):
    labels = _get_labels()
    sorted_list = sorted(labels, key=lambda x: getattr(x, attribute), reverse=False)
    return [label.name for label in sorted_list if not label.ignoreInEval]
