import mlflow.pytorch
import os
from PIL import Image
from tqdm import trange
from torchvision import transforms
import torch
from pathlib import Path
import numpy as np


def set_mlflow_tracking_uri(logs_path):
    mlflow.set_tracking_uri(f"file://{str(logs_path)}")


def load_model(run_id):
    model_uri = f"runs:/{run_id}/best_model"
    return mlflow.pytorch.load_model(model_uri)


def load_and_preprocess_image(img_path, preprocess):
    image = Image.open(img_path).convert("RGB")
    return preprocess(image)


def load_and_preprocess_images(image_dir):
    preprocess = transforms.Compose([transforms.ToTensor()])
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_dir, filename)
            image = load_and_preprocess_image(img_path, preprocess)
            images.append(image)
    return images


def infer_batch(model, batch, device):
    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.sigmoid(outputs)
        predictions = probabilities.round().int()
    return predictions.squeeze().cpu().tolist()


def infer_images(model, images, batch_size=1024, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    all_predictions = []
    model.to(device)
    for i in trange(0, len(images), batch_size):
        batch = torch.stack(images[i : i + batch_size]).to(device)
        predictions = infer_batch(model, batch, device)
        all_predictions.extend(predictions)
    return all_predictions


def ensemble_inference(models, images, batch_size=1024, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    all_predictions = []
    for model in models:
        predictions = infer_images(model, images, batch_size, device)
        all_predictions.append(predictions)
    return np.mean(all_predictions, axis=0).round().astype(int).tolist()


def save_predictions_to_file(predictions, file_path):
    with open(file_path, "w") as file:
        for pred in predictions:
            file.write(f"{pred}\n")


def main(run_ids, image_dir, output_file):
    # Set up MLflow tracking
    logs_path = Path.cwd() / "logs" / "mlruns"
    set_mlflow_tracking_uri(logs_path)

    # Load models
    print("Loading models...")
    models = [load_model(run_id) for run_id in run_ids]
    print(f"{len(models)} models loaded!")

    # Load and preprocess images
    print("Loading images...")
    images = load_and_preprocess_images(image_dir)
    print(f"{len(images)} images loaded and preprocessed.")

    # Perform ensemble inference
    print("Performing ensemble inference...")
    predictions = ensemble_inference(models, images)

    # Save predictions to file
    save_predictions_to_file(predictions, output_file)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    RUN_IDS = ["7f6fb4cf3c664dec9831facf82a3938a"]
    IMAGE_DIR = "./data/val_img"
    OUTPUT_FILE = f"predictions/ensemble_predictions_resnet18_adam_wBCE.txt"

    main(RUN_IDS, IMAGE_DIR, OUTPUT_FILE)
