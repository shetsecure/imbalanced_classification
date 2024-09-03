import yaml
from pathlib import Path
from train_model import train_model


def run_experiment(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    xp_name = Path(config_file).stem
    train_model(config, xp_name)


if __name__ == "__main__":
    configs = [
        "configs/resnet_focal.yaml",
        "configs/resnet_bce.yaml",
        "configs/resnet_weighted_bce_sgd.yaml",
        "configs/resnet_weighted_bce_adam.yaml",
    ]
    n_runs = 5

    print(f"Will run {n_runs} runs per xp")

    for config_file in configs:
        for _ in range(n_runs):
            print(f"Running experiment with config: {config_file}")
            run_experiment(config_file)
