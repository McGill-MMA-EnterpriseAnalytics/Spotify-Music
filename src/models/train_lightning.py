import os

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
import joblib  # Import joblib for saving the model


class ClassificationModel(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_size, num_hidden_layers, dropout_rate, activation, lr):
        super().__init__()
        self.save_hyperparameters()
        layers = [nn.Linear(input_size, hidden_size),
                  get_activation_fn(activation)()]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation_fn(activation)())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "elu":
        return nn.ELU
    else:
        raise ValueError(f"Invalid activation function: {activation}")


if __name__ == "__main__":
    # Load your data
    X_train_transformed, X_test_transformed, y_train, y_test = joblib.load(
        "./data/processed/train_test_transformed_data.joblib")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(
        X_train_transformed.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_test_transformed.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test.values, dtype=torch.long)
    # Create PyTorch datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create PyTorch data loaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_int("hidden_size", 32, 256, step=16)
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        activation = trial.suggest_categorical(
            "activation", ["relu", "leaky_relu", "elu"])

        model = ClassificationModel(input_size=X_train_transformed.shape[1], num_classes=len(np.unique(y_train)),
                                    hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
                                    dropout_rate=dropout_rate, activation=activation, lr=lr)

        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=5, verbose=True, mode="min")
        trainer = pl.Trainer(max_epochs=100, callbacks=[
                             early_stop_callback], accelerator="cuda" if torch.cuda.is_available() else "auto")

        trainer.fit(model, train_loader, val_loader)

        return trainer.callback_metrics["val_loss"].item()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_trial.params
    print("Best hyperparameters: ", best_params)
