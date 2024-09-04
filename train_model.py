import torch
import torch.optim as optim

import mlflow
import mlflow.pytorch

from pathlib import Path
from tqdm import tqdm

from dataset import load_dataset, create_dataloaders
from utils.common_utils import calculate_metrics, parse_classification_report, calculate_hter
from models.model_factory import get_model
from losses.loss_factory import get_loss_function
from samplers.progressive_sampler import ProgressiveSampler
from sklearn.metrics import classification_report


def setup_experiment(config, xp_name=None):
    """Set up MLflow experiment and log parameters."""

    logs_path = Path.cwd() / config["mlflow"]["log_dir"]
    mlflow.set_tracking_uri(f"file://{str(logs_path)}")

    experiment_name = xp_name if xp_name else "Thera Classification Experiments"
    mlflow.set_experiment(experiment_name)

    mlflow.start_run()

    mlflow.set_tag("experiment_type", config["mlflow"]["experiment_type"])
    mlflow.set_tag("model", config["model"]["name"])

    mlflow.log_params(config["training"])
    mlflow.log_params(config["optimizer"]["params"])
    mlflow.log_params({"model": config["model"]["name"]})
    mlflow.log_params({"loss_function": config["loss_function"]["type"]})


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


def load_model_and_optimizer(config):
    """Load the model and optimizer based on the configuration."""

    model = get_model(config)
    model = model.to(config["device"])

    optimizer = get_optimizer(config, model)
    return model, optimizer


def train_one_epoch(model, train_loader, criterion, optimizer, config):
    """Train the model for one epoch."""

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

    return running_loss / len(train_loader)


def validate_model(model, val_loader, criterion, config):
    """Validate the model and return validation loss and HTER."""

    val_loss, all_labels, all_preds = calculate_metrics(model, val_loader, criterion, config["device"])

    hter = calculate_hter(all_labels, all_preds)

    return val_loss, hter, all_labels, all_preds


def save_best_model(model, best_hter, current_hter, best_model_path):
    """Save the model if it has the lowest HTER so far."""

    if current_hter < best_hter:
        torch.save(model.state_dict(), best_model_path)
        mlflow.log_artifact(best_model_path, artifact_path="best_model")
        return current_hter
    return best_hter


def log_metrics(epoch, running_loss, val_loss, hter, all_labels, all_preds):
    """Log training and validation metrics to MLflow."""

    mlflow.log_metric("train_loss", running_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("HTER", hter, step=epoch)

    report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"])
    metrics = parse_classification_report(report)

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value, step=epoch)


def train_model(config, xp_name=None):
    """Main training function."""

    best_hter = float("inf")
    best_model_path = "best_model.pth"

    setup_experiment(config, xp_name)

    mlflow.log_dict(config, "config.yaml")

    train_dataset, val_dataset = load_dataset(config)

    if config["training"].get("use_progressive_sampling", False):
        sampler = ProgressiveSampler(
            train_dataset.labels,
            start_ratio=config["training"]["start_ratio"],
            end_ratio=config["training"]["end_ratio"],
            num_epochs=config["training"]["n_epochs"],
        )
        train_loader, val_loader = create_dataloaders(config, train_dataset, val_dataset, sampler=sampler)
    else:
        train_loader, val_loader = create_dataloaders(config, train_dataset, val_dataset)

    train_labels = torch.cat([labels for _, labels in train_loader]).cpu().numpy()
    config["loss_function"]["train_labels"] = train_labels

    model, optimizer = load_model_and_optimizer(config)
    criterion = get_loss_function(config)

    try:
        for epoch in tqdm(range(config["training"]["n_epochs"]), desc="Training Progress"):
            if config["training"].get("use_progressive_sampling", False):
                sampler.set_epoch(epoch)

            running_loss = train_one_epoch(model, train_loader, criterion, optimizer, config)
            val_loss, hter, all_labels, all_preds = validate_model(model, val_loader, criterion, config)

            print(
                f"Epoch [{epoch+1}/{config['training']['n_epochs']}], Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, HTER: {hter:.4f}"
            )
            log_metrics(epoch, running_loss, val_loss, hter, all_labels, all_preds)

            best_hter = save_best_model(model, best_hter, hter, best_model_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")

    finally:
        # Load the best model weights before logging to MLflow
        model.load_state_dict(torch.load(best_model_path))

        # Log the final model with an input example to generate a model signature
        with torch.no_grad():
            val_images, _ = next(iter(val_loader))
            val_images = val_images.cpu().numpy()
            model = model.to(torch.device("cpu"))
            mlflow.pytorch.log_model(model, "final_model", input_example=val_images)

        mlflow.end_run()
