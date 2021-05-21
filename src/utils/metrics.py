from collections import defaultdict, OrderedDict
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from scipy.optimize import linear_sum_assignment
from scipy.special import comb

from utils.logger import print_warning


def _comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)


class Metrics:
    def __init__(self, *names):
        self.names = list(names)
        self.curves = defaultdict(list)
        self.meters = defaultdict(AverageMeter)

    def reset(self, *names):
        if len(names) == 0:
            names = self.names
        for name in names:
            self.meters[name].reset()

    def __getitem__(self, name):
        return self.meters[name]

    def __repr__(self):
        return ', '.join(['{}={:.4f}'.format(name, self.meters[name].avg) for name in self.names])

    @property
    def avg_values(self):
        return [self.meters[name].avg for name in self.names]

    def update(self, *name_val):
        if len(name_val) == 1:
            d = name_val[0]
            assert isinstance(d, dict)
            for k, v in d.items():
                self.update(k, v)
        else:
            name, val = name_val
            if name not in self.names:
                self.names.append(name)
            if isinstance(val, (tuple, list)):
                assert len(val) == 2
                self.meters[name].update(val[0], n=val[1])
            else:
                self.meters[name].update(val)


class Scores:
    """
    Compute the following scores:
        - nmi
        - nmi diff (nmi computed with previous assignements)
        - global accuracy
        - mean accuracy
        - accuracy by class
    """
    def __init__(self, n_classes, n_clusters, linear_mapping=True):
        self.n_classes = n_classes
        self.n_clusters = n_clusters
        self.n_max_labels = max(n_classes, n_clusters)
        self.names = ['nmi', 'nmi_diff', 'global_acc', 'avg_acc'] + [f'acc_cls{i}' for i in range(n_classes)]
        self.values = OrderedDict(zip(self.names, [0] * len(self.names)))
        self.score_name = 'nmi'
        self.prev_labels_pred = None
        self.linear_mapping = linear_mapping
        self.reset()

    def compute(self):
        nmi = nmi_score(self.labels_true, self.labels_pred, average_method='arithmetic')
        if self.prev_labels_pred is not None:
            nmi_diff = nmi_score(self.prev_labels_pred, self.labels_pred, average_method='arithmetic')
        else:
            nmi_diff = 0

        matrix = self.compute_confusion_matrix()
        acc = np.diag(matrix).sum() / matrix.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_by_class = np.diag(matrix) / matrix.sum(axis=1)
        avg_acc = np.mean(np.nan_to_num(acc_by_class))
        self.values = OrderedDict(zip(self.names, [nmi, nmi_diff, acc, avg_acc] + acc_by_class.tolist()))
        return self.values

    def __getitem__(self, k):
        return self.values[k]

    def reset(self):
        if hasattr(self, 'labels_pred'):
            self.prev_labels_pred = self.labels_pred
        self.labels_true = np.array([], dtype=np.int64)
        self.labels_pred = np.array([], dtype=np.int64)

    def update(self, labels_true, labels_pred):
        self.labels_true = np.hstack([self.labels_true, labels_true.flatten()])
        self.labels_pred = np.hstack([self.labels_pred, labels_pred.flatten()])

    def compute_confusion_matrix(self):
        # XXX 100x faster than sklearn.metrics.confusion matrix, returns matrix with GT as rows, pred as columns
        matrix = np.bincount(self.n_max_labels * self.labels_true + self.labels_pred,
                             minlength=self.n_max_labels**2).reshape(self.n_max_labels, self.n_max_labels)
        matrix = matrix[:self.n_classes, :self.n_clusters]
        if self.n_clusters == self.n_classes:
            if self.linear_mapping:
                # we find the best 1-to-1 assignent with the Hungarian algo
                best_assign_idx = linear_sum_assignment(-matrix)[1]
                matrix = matrix[:, best_assign_idx]
        else:
            # we assign each cluster to its argmax class and aggregate clusters corresponding to the same class
            # TODO improve when argmax reached for several indices
            indices = np.argmax(matrix, axis=0)
            matrix = np.vstack([matrix[:, indices == k].sum(axis=1) for k in range(self.n_classes)]).transpose()
        return matrix


class SegmentationScores:
    """
    Compute the following metrics:
        - global accuracy (with and wo bkg class)
        - average accuracy by class (with and wo bkg class)
        - average IoU (with and wo_bkg class)
        - IoU by class
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes  # XXX should take a background class into account
        self.names = ["global_acc_nobkg", "avg_acc_nobkg", "avg_iou_nobkg", "global_acc", "avg_acc", "avg_iou"] \
            + [f'iou_cls{k}' for k in range(n_classes)]
        self.score_name = "avg_iou"
        self.reset()
        self.values = OrderedDict(zip(self.names, [0] * len(self.names)))

    def __getitem__(self, k):
        return self.values[k]

    def compute(self):
        metrics = []
        for withbkg in [False, True]:
            hist = self.confusion_matrix[1:, 1:] if not withbkg else self.confusion_matrix
            best_assign_idx = linear_sum_assignment(-hist)[1]
            hist = hist[:, best_assign_idx]
            global_acc = np.diag(hist).sum() / hist.sum()
            acc = np.diag(hist) / hist.sum(axis=1)
            avg_acc = np.mean(np.nan_to_num(acc))
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            avg_iu = np.mean(np.nan_to_num(iu))
            metrics += [global_acc, avg_acc, avg_iu]

        metrics += list(iu)
        self.values = OrderedDict(zip(self.names, metrics))
        return self.values

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def update(self, label_true, label_pred):
        lt_flat, lp_flat = label_true.flatten(), label_pred.flatten()
        self.confusion_matrix += self._fast_hist(lt_flat, lp_flat)

    def _fast_hist(self, label_true, label_pred):
        hist = np.bincount(self.n_classes * label_true + label_pred,
                           minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        return hist


class InstanceSegScores:
    """Computes Adjusted Rand Index, original implementation from sklearn.metrics.adjusted_rand_score"""
    def __init__(self, n_instances, with_bkg=False):
        self.n_instances = n_instances  # XXX should take a background instance into account
        self.names = ["mean_ari"]
        self.score_name = "mean_ari"
        self.with_bkg = with_bkg
        self.reset()
        self.values = OrderedDict(zip(self.names, [0] * len(self.names)))

    def reset(self):
        self.aris = []

    def __getitem__(self, k):
        return self.values[k]

    def compute(self):
        self.values = OrderedDict(zip(self.names, [np.mean(self.aris)]))
        return self.values

    def update(self, label_true, label_pred):
        B = len(label_true)
        for k in range(B):
            ari = self.cpu_ari(label_true[k], label_pred[k])
            self.aris.append(ari)

    def cpu_ari(self, label_true, label_pred):
        label_true, label_pred = label_true.flatten(), label_pred.flatten()
        if not self.with_bkg:
            # we remove background gt pixels from the computation
            good_idx = label_true != 0
            label_true, label_pred = label_true[good_idx], label_pred[good_idx]
        confusion_matrix = self._fast_hist(label_true, label_pred)
        sum_comb_c = sum(_comb2(n_c) for n_c in np.ravel(confusion_matrix.sum(axis=1)))
        sum_comb_k = sum(_comb2(n_k) for n_k in np.ravel(confusion_matrix.sum(axis=0)))
        sum_comb_table = sum(_comb2(n_ij) for n_ij in confusion_matrix.flatten())
        sum_comb_n = _comb2(confusion_matrix.sum())
        if (sum_comb_c == sum_comb_k == sum_comb_n == sum_comb_table):
            return 1.0
        else:
            prod_comb = (sum_comb_c * sum_comb_k) / sum_comb_n
            mean_comb = (sum_comb_k + sum_comb_c) / 2.
            return (sum_comb_table - prod_comb) / (mean_comb - prod_comb)

    def _fast_hist(self, label_true, label_pred):
        try:
            hist = np.bincount(self.n_instances * label_true + label_pred,
                               minlength=self.n_instances ** 2).reshape(self.n_instances, self.n_instances)
        except ValueError:
            print_warning('InstanceSegScores._fast_hist error: labels in GT are greater than nb instances')
            val = np.unique(label_true)[1:]
            new_label = np.zeros(label_true.shape, dtype=np.uint8)
            for k, v in enumerate(val):
                new_label[label_true == v] = k+1
            hist = np.bincount(self.n_instances * new_label + label_pred,
                               minlength=self.n_instances ** 2).reshape(self.n_instances, self.n_instances)
        return hist


class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class AverageTensorMeter:
    """AverageMeter for tensors of size (B, *dim) over B dimension"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.count = 0

    def update(self, t):
        n = t.size(0)
        if n > 0:
            avg = t.mean(dim=0)
            self.avg = (self.count * self.avg + n * avg) / (self.count + n)
            self.count += n
