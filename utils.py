import os
import torch
import shutil
from torch.nn import init
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import numpy as np

class AverageMeter(object):
    # Computes and stores the average and current value

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


def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    model_path = os.path.join(directory, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(directory, 'model_best_acc.pth.tar'))

def compute_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1)
    acc = (pred == label).type(torch.FloatTensor).mean().item()
    f1 = f1_score(pred, label, average='micro')
    # uar = recall_score(pred, label, average='macro')
    return acc, f1

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def sanity_check(labels, n_samples):
    for label in labels:
        if len(class_sample_indices[key]) < n_samples:
            raise ValueError('There is a class with less samples than n_support + n_query. The label was %s' % label)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def euclidean_dist(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def accuracy_top_k(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def point_distance(point1, point2, metric="EUCLIDEAN"):
    if metric == "EUCLIDEAN":
        return np.sqrt(np.sum((point1 - point2)**2))

def remvove_doubles(points):

    new_points = []
    for point in points:
        min_dist = np.inf
        for i in range(len(new_points)):
            dist = point_distance(point, new_points[i])
            if dist < min_dist:
                min_dist = dist
        if min_dist > 1e-10:
            new_points.append(point)
    new_points = np.array(new_points)
    return new_points

def get_distance_matrix(points1, points2):
    length_1 = len(points1)
    length_2 = len(points2)
    dist_matrix = np.empty((length_1, length_2))
    for i in range(length_1):
        for j in range(length_2):
            dist_matrix[i,j] = point_distance(points1[i], points2[j])
    return dist_matrix

def find_minimal_average_distance_betwen_points(dist_matrix):
    length = len(dist_matrix)
    total_distance = 0
    for i in range(length):
        dist_row = dist_matrix[i, :]
        min_idx = np.argmin(dist_row)
        min_dist = dist_row[min_idx]
        total_distance += min_dist

    return total_distance/length