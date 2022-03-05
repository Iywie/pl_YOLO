from pytorch_lightning import LightningModule
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss
import torch
import numpy as np

from models.backbones.resnet import ResNet
from models.backbones.darknet_csp import CSPDarkNet
from models.heads.cls_head import ClsHead


class LitResnet(LightningModule):

    def __init__(self, cfgs):
        super().__init__()

        self.cfg_dataset = cfgs['dataset']

        self.data_dir = self.cfg_dataset['dir']
        self.batch_size = self.cfg_dataset['train_batch_size']
        self.num_classes = 10

        self.num_workers = 4
        self.pin_mem = True
        # self.backbone = ResNet(num_classes=self.num_classes)
        # self.head = ClsHead(512, num_classes=self.num_classes)
        self.backbone = CSPDarkNet(out_features=["dark5"])
        self.head = ClsHead(256, num_classes=self.num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        result = self.backbone(x)
        result = self.head(result)
        loss = get_loss(self.num_classes, result, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        result = self.backbone(x)
        result = self.head(result)
        preds = torch.argmax(result, dim=-1)
        return preds, y

    def validation_epoch_end(self, val):
        all_preds = np.array([])
        all_labels = np.array([])
        for batch in val:
            all_preds = np.append(all_preds, batch[0].detach().cpu().numpy())
            all_labels = np.append(all_labels, batch[1].detach().cpu().numpy())
        accuracy = (all_preds == all_labels).mean()
        print("Val Accuracy: %2.5f" % accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=3e-2,
                                    momentum=0.9,
                                    weight_decay=0)
        return optimizer

    def train_dataloader(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.CIFAR10(self.data_dir, train=True, transform=transform_train)
        sampler_train = torch.utils.data.RandomSampler(dataset)
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
        )
        return data_loader_train

    def val_dataloader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.CIFAR10(self.data_dir, train=False, transform=transform_test)
        sampler_val = torch.utils.data.SequentialSampler(dataset)
        data_loader_val = torch.utils.data.DataLoader(
            dataset, sampler=sampler_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
        )
        return data_loader_val


normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.244, 0.262])


def get_loss(num_classes, result, labels):
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(result.view(-1, num_classes), labels.view(-1))
    return loss


def pred_labels(all_preds, all_label, logits, y):
    preds = torch.argmax(logits, dim=-1)
    all_preds = np.append(all_preds, preds.detach().cpu().numpy())
    all_label = np.append(all_label, y.detach().cpu().numpy())
    return all_preds, all_label
