import torch
import torch.nn as nn
import sys
sys.path.append('../../')
from model.loss import YoloLoss

# TODO: Add train function
# TODO: Add training script and test weather it works or not.

# def train_fn(model, train_loader, val_loader, test_loader, )


# def train_model(model, train_loader):
#     # Training
#     for img, label in train_loader:
#         outputs = model(img)
#         return outputs