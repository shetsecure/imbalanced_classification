import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from mlflow.models.evaluation import evaluate
import torchvision
from sklearn.metrics import classification_report
import numpy as np

from utils import calculate_metrics, parse_classification_report, calculate_hter, FocalLoss
from dataset import load_data

from pathlib import Path


# def get_loss_function(config):
#     """Dynamically get the loss function from config."""
#     if "BCEWithLogitsLoss" in config["loss_function"]["type"]:
#         if config["loss_function"].get("pos_weight"):
#             class_counts = np.bincount(config["loss_function"]["train_labels"])
#             class_weights = 1.0 / class_counts

#             # Apply custom scaling if specified
#             if config["loss_function"]["custom_weight_scale"]["use_custom"]:
#                 weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(config["device"])
#                 weights *= config["loss_function"]["custom_weight_scale"]["factor"]
#             else:
#                 weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(config["device"])
#                 weights *= config["loss_function"]["weight_factor"]

#             print(f"BCE weights {weights}")
#             return nn.BCEWithLogitsLoss(pos_weight=weights[1])
#         else:
#             return nn.BCEWithLogitsLoss()
#     else:
#         loss_class = getattr(nn, config["loss_function"]["type"])
#         return loss_class(**config["loss_function"]["params"])


# def log_loss_function_config(config):
#     """Log loss function configuration with MLflow."""
#     loss_function_config = config["loss_function"]
#     mlflow.log_param("loss_function_type", loss_function_config["type"])

#     if "BCEWithLogitsLoss" in config["loss_function"]["type"]:
#         if config["loss_function"].get("pos_weight"):
#             mlflow.log_param("loss_function_pos_weight", loss_function_config["pos_weight"])
#             mlflow.log_param("loss_function_weight_factor", loss_function_config["weight_factor"])
#             mlflow.log_param("loss_function_custom_weight_scale_use_custom", loss_function_config["custom_weight_scale"]["use_custom"])
#             mlflow.log_param("loss_function_custom_weight_scale_factor", loss_function_config["custom_weight_scale"]["factor"])

#     for key, value in loss_function_config.get("params", {}).items():
#         mlflow.log_param(f"loss_function_param_{key}", value)


def get_loss_function(config):
    """Dynamically get the loss function from config."""
    if config["loss_function"]["type"] == "BCEWithLogitsLoss":
        if config["loss_function"].get("pos_weight"):
            class_counts = np.bincount(config["loss_function"]["train_labels"])
            class_weights = 1.0 / class_counts

            if config["loss_function"]["custom_weight_scale"]["use_custom"]:
                weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(config["device"])
                weights *= config["loss_function"]["custom_weight_scale"]["factor"]
            else:
                weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(config["device"])
                weights *= config["loss_function"]["weight_factor"]

            print(f"BCE weights {weights}")
            return nn.BCEWithLogitsLoss(pos_weight=weights[1])
        else:
            return nn.BCEWithLogitsLoss()
    elif config["loss_function"]["type"] == "FocalLoss":
        return FocalLoss(
            alpha=config["loss_function"].get("alpha", 0.25),
            gamma=config["loss_function"].get("gamma", 2.0),
            reduction=config["loss_function"].get("reduction", "mean"),
        )
    else:
        loss_class = getattr(nn, config["loss_function"]["type"])
        return loss_class(**config["loss_function"]["params"])


def log_loss_function_config(config):
    """Log loss function configuration with MLflow."""
    loss_function_config = config["loss_function"]
    mlflow.log_param("loss_function_type", loss_function_config["type"])

    if loss_function_config["type"] == "FocalLoss":
        mlflow.log_param("loss_function_alpha", loss_function_config.get("alpha", 0.25))
        mlflow.log_param("loss_function_gamma", loss_function_config.get("gamma", 2.0))
        mlflow.log_param("loss_function_reduction", loss_function_config.get("reduction", "mean"))
    elif "BCEWithLogitsLoss" in loss_function_config["type"]:
        if config["loss_function"].get("pos_weight"):
            mlflow.log_param("loss_function_pos_weight", loss_function_config["pos_weight"])
            mlflow.log_param("loss_function_weight_factor", loss_function_config["weight_factor"])
            mlflow.log_param("loss_function_custom_weight_scale_use_custom", loss_function_config["custom_weight_scale"]["use_custom"])
            mlflow.log_param("loss_function_custom_weight_scale_factor", loss_function_config["custom_weight_scale"]["factor"])

    for key, value in loss_function_config.get("params", {}).items():
        mlflow.log_param(f"loss_function_param_{key}", value)


def get_optimizer(config, model):
    """Initialize optimizer based on the configuration."""
    optimizer_name = config["optimizer"]["name"]
    optimizer_params = config["optimizer"]["params"]

    optimizer_classes = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
    }

    if optimizer_name not in optimizer_classes:
        raise ValueError(f"Optimizer '{optimizer_name}' is not recognized. Available options: {list(optimizer_classes.keys())}")

    optimizer_class = optimizer_classes[optimizer_name]

    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    return optimizer


def train_model(config, xp_name=None):

    best_hter = float("inf")  # Initialize with a large number or use `-float('inf')` if higher is better
    best_model_path = "best_model.pth"

    train_loader, val_loader, train_labels = load_data(config)
    config["loss_function"]["train_labels"] = train_labels  # Pass labels for possible weighted loss

    # Initialize model
    model_name = config["model"]["name"]
    model = getattr(torchvision.models, model_name)(pretrained=True)

    # Adjust for binary classification
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, 1)  # ResNet, VGG, etc.
    elif hasattr(model, "classifier"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)  # For models like EfficientNet
    else:
        raise NotImplementedError(f"Model {model_name} is not supported for modification")

    model = model.to(config["device"])

    # Set up the loss function
    criterion = get_loss_function(config)

    # Initialize optimizer
    optimizer = get_optimizer(config, model)

    logs_path = Path.cwd() / config["mlflow"]["log_dir"]
    mlflow.set_tracking_uri(f"file://{str(logs_path)}")

    xp_name = xp_name if xp_name else "Thera Classification Experiments"
    mlflow.set_experiment(xp_name)

    mlflow.start_run()

    mlflow.set_tag("experiment_type", config["mlflow"]["experiment_type"])
    mlflow.set_tag("model", config["model"]["name"])

    mlflow.log_params(config["training"])
    log_loss_function_config(config)

    optimizer_params = config["optimizer"]["params"]
    mlflow.log_params({"optimizer_name": config["optimizer"]["name"], **optimizer_params})
    mlflow.log_params({"model": model_name})

    for epoch in range(config["training"]["n_epochs"]):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(config["device"]).float(), labels.to(config["device"]).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config['training']['n_epochs']}], Train Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        val_loss, all_labels, all_preds = calculate_metrics(model, val_loader, criterion, config["device"])
        hter = calculate_hter(all_labels, all_preds)

        # Log metrics with MLflow
        mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)
        mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)

        print(f"Epoch [{epoch+1}/{config['training']['n_epochs']}], Val Loss: {val_loss / len(val_loader):.4f}")
        mlflow.log_metric("HTER", hter, step=epoch)

        # Classification report and HTER calculation
        report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"])
        metrics = parse_classification_report(report)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)

        if hter < best_hter:
            best_hter = hter
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(best_model_path, artifact_path="best_model")

    # Log the final model with an input example to generate a model signature
    with torch.no_grad():
        val_images, _ = next(iter(val_loader))
        val_images = val_images.cpu().numpy()
        model = model.to(torch.device("cpu"))
        mlflow.pytorch.log_model(model, "final_model", input_example=val_images)

    mlflow.end_run()
