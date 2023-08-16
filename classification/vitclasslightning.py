import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from src.models.vision_transformer import VisionTransformer, vit_predictor, vit_small, MLP
from src.helper import init_model, load_checkpoint, init_opt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
from lightning.pytorch.loggers import WandbLogger

class IjepaCifar10(pl.LightningModule):
    def __init__(self, checkpoint_path = '/home/christina/PycharmProjects/ijepa/logs/jepa-latest.pth.tar', num_classes=10, learning_rate=1e-3, track_wandb=True):
        super().__init__()
        checkpoint = torch.load(checkpoint_path)
        self.encoder, self.predictor = init_model(
            device='cuda',
            patch_size=16,
            model_name='vit_small', #changed from vit_base
            crop_size=224,
            pred_depth=6,
            pred_emb_dim=384
        )
        #self.target_encoder = copy.deepcopy(encoder)
        def removeprefix(checkpointkey):
            new_state_dict = {}
            for key, value in checkpointkey.items():
                if key.startswith('module.'):
                    new_key = key[len('module.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict

        self.encoder.load_state_dict(removeprefix(checkpoint['encoder']))

        def remove_keys_between_indices(dictionary, start_index, end_index):
            new_dict = {k: v for i, (k, v) in enumerate(dictionary.items()) if i < start_index or i > end_index}
            return new_dict

        pred_state_dict = removeprefix(checkpoint['predictor'])
        new_pred_state_dict = remove_keys_between_indices(pred_state_dict, 76, 147)
        self.predictor.load_state_dict(new_pred_state_dict)

        #self.target_encoder.load_state_dict(removeprefix(checkpoint['target_encoder']))
        classification_head = nn.Linear(in_features=self.predictor.embed_dim, out_features=10)
        self.predictor = classification_head
        self.learning_rate = learning_rate

        self.track_wandb = track_wandb
        self.train_step_losses = []
        self.val_step_losses = []
        self.train_step_acc = []
        self.val_step_acc = []
        self.last_train_acc = 0
        self.last_train_loss = 0

    def forward(self,x):
        enc_output = self.encoder(x)
        pred_output = self.predictor(enc_output)
        pred_output_pooled = torch.mean(pred_output, dim=1)
        return pred_output_pooled

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        enc_output = self.encoder(images.to('cuda'))
        pred_output = self.predictor(enc_output)
        pred_output_pooled = torch.mean(pred_output, dim=1)
        loss = nn.CrossEntropyLoss()(pred_output_pooled, labels)
        self.train_step_losses.append(loss)
        # log accuracy
        _, preds = torch.max(pred_output_pooled.data, dim=1)
        acc = (preds == labels).sum().item() / pred_output_pooled.size(dim=0)
        acc *= 100
        self.train_step_acc.append(acc)

        return {'loss': loss}

    def on_train_epoch_end(self):
        all_preds = self.train_step_losses
        avg_loss = sum(all_preds) / len(all_preds)

        all_acc = self.train_step_acc
        avg_acc = sum(all_acc) / len(all_acc)
        avg_acc = round(avg_acc, 2)

        self.last_train_loss = avg_loss
        self.last_train_acc = avg_acc

        self.train_step_acc.clear()
        self.train_step_losses.clear()

        return {'train_loss': avg_loss, 'train_acc': avg_acc}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        enc_output = self.encoder(images.to('cuda'))
        pred_output = self.predictor(enc_output)
        pred_output_pooled = torch.mean(pred_output, dim=1)
        loss = nn.CrossEntropyLoss()(pred_output_pooled, labels)
        self.val_step_losses.append(loss)
        # log accuracy
        _, preds = torch.max(pred_output_pooled.data, dim=1)
        acc = (preds == labels).sum().item() / pred_output_pooled.size(dim=0)
        acc *= 100
        self.val_step_acc.append(acc)

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        all_preds = self.val_step_losses
        all_acc = self.val_step_acc

        avg_loss = sum(all_preds) / len(all_preds)
        avg_acc = sum(all_acc) / len(all_acc)
        # avg_acc = round(avg_acc, 2)

        if self.track_wandb:
            wandb.log({"training_loss": self.last_train_loss,
                       "training_acc": self.last_train_acc,
                       "validation_loss": avg_loss,
                       "validation_acc": avg_acc})

        self.val_step_losses.clear()
        self.val_step_acc.clear()

        return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("test_loss", loss)


if __name__ == "__main__":
    # pl.seed_everything(42)


    wandb.init(project="ijepa-classification")
    wandb_logger = WandbLogger(project="ijepa-classification")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


    model = IjepaCifar10()


    trainer = pl.Trainer(num_nodes=1, max_epochs=50, logger=wandb_logger)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model on val set
    # trainer.test(model, dataloaders=val_loader)
    # trainer.validate(model, dataloaders=val_loader)

