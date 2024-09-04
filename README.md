# imbalanced_classification

## High level view of project structure
```
├── best_model.pth
├── configs
│   ├── efficientnet_bce.yaml
│   ├── mobilenet_bce.yaml
│   ├── progressive_resnet.yaml
│   ├── resnet_bce.yaml
│   ├── resnet_focal.yaml
│   ├── resnet_weighted_bce_adam.yaml
│   └── resnet_weighted_bce_sgd.yaml
├── data
│   ├── label_train.txt
│   ├── train_img
│   └── val_img
├── dataset.py
├── Exercice Machine Learning TheraPanacea.pdf
├── infer.py
├── logs
│   ├── mlruns
│   └── models
├── losses
│   ├── __init__.py
│   ├── loss_factory.py
├── models
│   ├── __init__.py
│   ├── model_factory.py
├── notebooks
│   ├── exploration.ipynb
│   └── train.ipynb
├── predictions
│   └── 6ebb4c2b577b4991931642f56a1ce9e4_predictions.txt
├── __pycache__
│   ├── dataset.cpython-311.pyc
│   └── train_model.cpython-311.pyc
├── README.md
├── run_experiments.py
├── samplers
│   ├── __init__.py
│   ├── progressive_sampler.py
├── train_model.py
└── utils
    ├── common_utils.py
    ├── __init__.py

```

Main script is: run_experiments.py

The code is organized in such way that we only change or create config files in order to execute a new experiment.
It is also done in a modular and generalizable way that allow us to choose any model from torchvision zoo to finetune.

You can find some examples in the configs directory.


## Problem statement

We have a binary classification problem of RGB 64x64 images.
We don't know exactly the significance of the labels that were provided.

The dataset is super imbalanced as shown in exploration notebook in the notebooks folder, so this is going to be the main issue.

## Data

Data already seems to be cleaned, so no pre-processing was done really.

Added random horizontal flipping to all xps for data augmentation.

## Experiment design

Each experiment was done multiple times (runs), which is equivalent to do a k-fold cross validation.

Each run, we split the provided dataset into a train/val sets in a <b>stratified manner</b>.

We log and keep track of all our xps using <b>MLflow</b>. I chose it since it also makes it easier to deploy the final model, or serve it, simulating a more realistic scenario.

## Ideas that I tried

- Simple classifier finetuning with BCE:
Just establishing the baseline

- Optimizing different classifiers with weighted BCE instead with a ration of 1 to 10 yielded much better performances. Resnet18 seems to be the best choice out of the different classifiers that were tried.

- Progressive sampling: involves gradually increasing the proportion of minority class samples in each training batch over time. -> Didn't really work that well.

- Focal loss: didn't work really

- One class learning: Shifting our perspective from binary classification, to one class learning. Basically treating our problem as anomaly detection instead, where we treat the minority class as an anomaly leverging the big number of examples we have in the majority class. -> Didn't give better results, but wasn't really explored that much due to time constraints.


## Remarks

In multiple experiments, especially whilst using Adam with a relatively big LR, we can notice that the validation loss starts to go up quickly after epoch 2 or 3.

The rapid increase in validation loss suggests overfitting, due to the class imbalance and maybe suboptimal regularization as well.


# Model selection

Model selection was based on HTER as requested, which is super correlated with the other metrics like val loss.

# Inference

Finetuning a resnet18 using Adam and a weighted BCE yielded the best results. 
I took multiple models from multiple runs with different f1-score of the minority class and some with the best HTER score to do the final prediction.

