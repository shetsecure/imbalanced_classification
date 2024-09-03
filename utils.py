import torch
import numpy as np


def calculate_metrics(model, val_loader, criterion, device):
    """
    Calculate metrics for a given model on the validation set.

    Args:
        model: The neural network model to evaluate.
        val_loader: DataLoader for the validation set.
        criterion: The loss function used for evaluation.
        device: Device on which the model is evaluated.

    Returns:
        Tuple of validation loss, numpy array of all true labels, and numpy array of all predicted labels.
    """
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device).float()
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            predicted = probs.round()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return val_loss, np.asarray(all_labels), np.asarray(all_preds)


def calculate_hter(all_labels, all_preds):
    # Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)

    false_acceptance = np.sum((all_labels == 0) & (all_preds == 1))
    false_rejection = np.sum((all_labels == 1) & (all_preds == 0))
    total_acceptance = np.sum(all_labels == 0)
    total_rejection = np.sum(all_labels == 1)

    far = false_acceptance / total_acceptance
    frr = false_rejection / total_rejection

    # Calculate the Half-Total Error Rate (HTER)
    hter = (far + frr) / 2
    return hter


def parse_classification_report(report):
    """Parse classification report and log metrics."""
    lines = report.split("\n")
    metrics = {}

    # Iterate over lines to capture metrics
    for line in lines[2:-3]:  # Skip header and last two lines
        if line.strip() == "":
            continue
        parts = line.split()

        # Check if the line has at least 5 parts (label + 4 metrics)
        if len(parts) >= 5:
            # Extract label which could be multiple words
            label = " ".join(parts[:-4])
            precision = float(parts[-4])
            recall = float(parts[-3])
            f1_score = float(parts[-2])

            # Log metrics with label
            metrics[f"{label}_precision"] = precision
            metrics[f"{label}_recall"] = recall
            metrics[f"{label}_f1_score"] = f1_score

    return metrics


import torch
import torch.nn as nn
import torch.nn.functional as F


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
