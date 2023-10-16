import numpy as np


def loss(preds, targets):
    # categorical crossentropy loss
    if len(preds) != len(targets):
        raise ValueError("Number of predictions and targets must be the same.")

    loss = 0
    for pred, target in zip(preds, targets):
        # pred is a 1D array of class probabilities
        # Calculate the categorical cross-entropy for each class
        class_loss = -np.log(pred[target])

        loss += class_loss

    return loss


def accuracy(preds, targets):
    if len(preds) != len(targets):
        raise ValueError("Number of predictions and targets must be the same.")

    correct = 0
    for pred, target in zip(preds, targets):
        pred = pred.argmax()
        if pred == target:
            correct += 1

    return correct / len(targets)