import torch.nn as nn
import torchvision.models as models


def get_model(config):
    """
    Create a model based on the configuration provided.

    Args:
        config (dict): A dictionary containing the model configuration.

    Returns:
        torch.nn.Module: The modified model for binary classification.
    """
    model_name = config["model"]["name"]
    model = getattr(models, model_name)(pretrained=True)

    # Adjust the model for binary classification
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, 1)  # ResNet, VGG, etc.
    elif hasattr(model, "classifier"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)  # For models like EfficientNet
    else:
        raise NotImplementedError(f"Model {model_name} is not supported for modification")

    return model
