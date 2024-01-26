"""
    Example training with ul_trainer
"""

import os
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ul_trainer import UlTrainer
from ul_trainer.loggers import TBLogger
from ul_trainer.loggers import TopKLogger
from ul_trainer import UlModule


class MyTrainDataset(Dataset):
    """
        Dummy dataset
    """

    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(10), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]


class UlMLP(UlModule):
    """
     MLP Model that extends the ULTModule
    """

    def __init__(self) -> None:
        super().__init__()
        layers = [nn.Linear(10, 1000)]
        layers.append(nn.ReLU())
        layers.append(nn.Linear(1000, 1))
        self.global_step = 0
        self.model = nn.Sequential(*layers)

    def training_step(self, batch, batch_idx=0):
        """
            Training Step
        """
        x, y = batch[0], batch[1]
        pred = self.model(x)
        loss = F.mse_loss(pred, y)
        logs = {"train/psnr": -1*loss.item()}
        return loss, logs

    @torch.no_grad()
    def validation_step(self, batch, batch_idx=0):
        """
            Training Step
        """
        self.model.eval()
        with torch.no_grad():
            x, y = batch[0], batch[1]
            pred = self.model(x)
            loss = F.mse_loss(pred, y)
            scalar_logs = {"val/psnr": -1*loss.item()}
            image_logs = {}
            image_logs["val/pred"] = torch.rand(1, 3, 32, 32)
            image_logs["val/gr"] = torch.rand(1, 3, 32, 32)
        self.model.train()
        return {"scalar": scalar_logs, "image": image_logs}

    def train_dataloader(self):
        """
            Training Data Loader
        """
        dataset = MyTrainDataset(100000)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset,
                          drop_last=False,
                          batch_size=64,
                          pin_memory=True,
                          shuffle=False,
                          sampler=sampler), sampler

    def val_dataloader(self):
        """
            Val Data Loader
        """
        dataset = MyTrainDataset(100)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset,
                          drop_last=False,
                          batch_size=64,
                          pin_memory=True,
                          shuffle=False,
                          sampler=sampler), sampler

    def configure_optimizers(self):
        """
            Configure optimizers
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0004)
        return [optimizer], [None]

if __name__ == "__main__":    
    log_dir = "./logs/exp_008"
    tb_logger = TBLogger(logdir=os.path.join(log_dir, "tensor_board"))
    topk_logger = TopKLogger(topk=2, logdir=os.path.join(log_dir, "topk"), monitor="val/psnr", monitor_mode="max")
    ult_model = UlMLP()

    trainer = UlTrainer(
        model=ult_model, topk_logger=topk_logger, tb_logger=tb_logger,
        validate_every_x_epoch=1,
        max_epochs=40,
        log_root=log_dir,
        tb_log_steps=2,
        use_ema=True,
        ema_update_interval=5)
    
    trainer.train()
