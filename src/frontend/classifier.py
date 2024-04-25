import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        num_classes,
        hidden_size,
        num_hidden_layers,
        dropout_rate,
        activation,
        lr,
    ):
        super().__init__()
        self.save_hyperparameters()
        layers = [nn.Linear(input_size, hidden_size), get_activation_fn(activation)()]
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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

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
