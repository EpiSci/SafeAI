# Copyright (c) 2018 Episys Science, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

def get_inout_stats(in_probs, out_probs, labels_in=None):

    stats = {}
    if labels_in is not None:
        predicted_labels = np.argmax(in_probs, axis=1)
        labels_in = np.array(labels_in).squeeze()
        assert predicted_labels.shape == labels_in.shape
        stats['accuracy'] = np.sum(np.equal(predicted_labels, labels_in))\
                                  / float(len(labels_in))

    in_probs_max = np.max(in_probs, axis=1)
    out_probs_max = np.max(out_probs, axis=1)
    trues = np.append(np.ones(len(in_probs)), np.zeros(len(out_probs)))
    trues_flipped = np.append(np.zeros(len(in_probs)), np.ones(len(out_probs)))
    probs = np.append(in_probs_max, out_probs_max)
    fpr, tpr, thresholds = roc_curve(trues, probs)
    corrects_by_thresh = [len(in_probs_max[in_probs_max > thr])
                          + len(out_probs_max[out_probs_max < thr])
                          for thr in thresholds]

    stats['avg_in_max_softmax'] = np.mean(in_probs_max)
    stats['avg_out_max_softmax'] = np.mean(out_probs_max)
    stats['auroc'] = roc_auc_score(trues, probs)
    stats['aupr-in'] = average_precision_score(trues, probs)
    stats['aupr-out'] = average_precision_score(trues_flipped, probs)
    stats['detection_accuracy'] = np.max(corrects_by_thresh) / float(len(probs))
    stats['fpr-at-tpr95'] = fpr[len(fpr) - len(tpr[tpr > 0.95])]

    return stats
