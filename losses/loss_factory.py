import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def get_loss_function(config):
    loss_type = config["loss_function"]["type"]

    if loss_type == "BCEWithLogitsLoss":
        if config["loss_function"].get("pos_weight"):
            class_counts = np.bincount(config["loss_function"]["train_labels"])
            class_weights = 1.0 / class_counts

            if config["loss_function"]["custom_weight_scale"]["use_custom"]:
                weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(config["device"])
                weights *= config["loss_function"]["custom_weight_scale"]["factor"]
            else:
                weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(config["device"])
                weights *= config["loss_function"]["weight_factor"]

            return nn.BCEWithLogitsLoss(pos_weight=weights[1])
        else:
            return nn.BCEWithLogitsLoss()

    elif loss_type == "FocalLoss":
        return FocalLoss(
            alpha=config["loss_function"].get("alpha", 0.25),
            gamma=config["loss_function"].get("gamma", 2.0),
            reduction=config["loss_function"].get("reduction", "mean"),
        )

    else:
        loss_class = getattr(nn, loss_type)
        return loss_class(**config["loss_function"]["params"])


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert targets to the same shape as inputs
        targets = targets.view(-1, 1)

        # Compute the binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Apply sigmoid to inputs to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate the focal weight
        targets = targets.type_as(inputs)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = at * torch.pow((1 - pt), self.gamma)

        # Apply the focal weight to the BCE loss
        focal_loss = focal_weight * BCE_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
