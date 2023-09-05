import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandomResizedCrop

from src.models.vision_transformer import VisionTransformer, vit_predictor, vit_small, MLP
from src.helper import init_model, load_checkpoint, init_opt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from transformers import ViTForImageClassification, AdamW, ViTModel
from vit_pytorch import SimpleViT
from pytorch_pretrained_vit import ViT
class IjepaCifar10(pl.LightningModule):
    def __init__(self, checkpoint_path = None, num_classes=10, learning_rate=1e-3, track_wandb=True):
        super().__init__()
        self.encoder, self.predictor = init_model(
            device='cuda',
            patch_size=16,
            model_name='vit_base',  # changed from vit_base
            crop_size=224,
            pred_depth=12,
            pred_emb_dim=768
        )
        if checkpoint_path is not None:

            checkpoint = torch.load(checkpoint_path)

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
            #self.predictor.load_state_dict(new_pred_state_dict)
        #pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        #self.encoder = torchvision.models.vit_b_16(weights=pretrained_weights)
        #self.encoder = ViT('B_16_imagenet1k', pretrained=True)
        #self.encoder = SimpleViT(   image_size = 224,
                                    # patch_size = 16,
                                    # num_classes = 10,
                                    # dim = 768,
                                    # depth = 12,
                                    # heads = 12,
                                    # mlp_dim = 3072)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        #self.encoder.fc = nn.Linear(in_features = self.encoder.fc.in_features, out_features=10)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.heads = nn.Linear(in_features = 768, out_features=10)
        lin_layer = nn.Linear(in_features = 768, out_features=10)
        classification_head = nn.Sequential(torch.nn.BatchNorm1d(768, affine=False, eps=1e-6),lin_layer)
        self.predictor = classification_head

        #freeze all features except for class head

        #
        # for param in self.predictor.parameters():
        #     param.requires_grad = True
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
        enc_output = enc_output.permute(0, 2, 1)
        pooled_output = self.avg_pooling(enc_output)
        pooled_output = pooled_output.view(pooled_output.size(0), -1)
        pred_output = self.predictor(pooled_output)
        pred_output = nn.functional.softmax(pred_output, dim=-1)
        return pred_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.LARS
        return optimizer

    def training_step(self, batch, batch_idx):

        images, labels = batch
        # enc_output = self.encoder(images.to('cuda'))
        # enc_output = enc_output.permute(0, 2, 1)
        # pooled_output = self.avg_pooling(enc_output)
        # pooled_output = pooled_output.view(pooled_output.size(0), -1)
        # pred_output = self.predictor(pooled_output)
        logits = self.forward(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.train_step_losses.append(loss)
        # log accuracy
        _, preds = torch.max(logits.data, dim=1)
        acc = (preds == labels).sum().item() / logits.size(dim=0)
        acc *= 100
        self.train_step_acc.append(acc)
        # if batch_idx == 0:
        #     wandb.log(images, preds,)
        #visualize the input, prediction, ground truth at batch idx 0
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
        # enc_output = self.encoder(images.to('cuda'))
        # enc_output = enc_output.permute(0, 2, 1)
        # pooled_output = self.avg_pooling(enc_output)
        # pooled_output = pooled_output.view(pooled_output.size(0), -1)
        # pred_output = self.predictor(pooled_output)
        # pred_output_pooled = torch.mean(pred_output, dim=1)
        logits = self.forward(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.val_step_losses.append(loss)
        # log accuracy
        _, preds = torch.max(logits.data, dim=1)
        acc = (preds == labels).sum().item() / logits.size(dim=0)
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
    #checkpoint= '/home/christina/PycharmProjects/ijepa/logs/jepa-latest.pth.tar'
    checkpoint = None
    wandb.init(project="ijepa-classification")
    wandb_logger = WandbLogger(project="ijepa-classification")

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    val_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


    model = IjepaCifar10( checkpoint_path = checkpoint)


    trainer = pl.Trainer(num_nodes=1, max_epochs=50, logger=wandb_logger)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model on val set
    # trainer.test(model, dataloaders=val_loader)
    # trainer.validate(model, dataloaders=val_loader)

