import numpy as np


def loss(preds, targets):
    # categorical crossentropy loss
    if len(preds) != len(targets):
        raise ValueError("Number of predictions and targets must be the same.")

    return -np.sum(targets * np.log(preds, where=(preds!=0)))


def loss_batch(preds_lst, targets_lst):
    # categorical crossentropy loss
    if len(preds_lst) != len(targets_lst):
        raise ValueError("Number of predictions and targets must be the same.")

    return sum(-np.sum(targets * np.log(preds, where=(preds!=0))) for preds, targets in zip(preds_lst, targets_lst))


def accuracy(preds, targets):
    if len(preds) != len(targets):
        raise ValueError("Number of predictions and targets must be the same.")

    correct = 0
    for pred, target in zip(preds, targets):
        pred = pred.argmax()
        if pred == target:
            correct += 1

    return correct / len(targets)