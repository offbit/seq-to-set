"""Helper functions."""


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_precision_recall(predictions, targets, k):
    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))
    return float(num_hit) / len(predictions), float(num_hit) / len(targets)
