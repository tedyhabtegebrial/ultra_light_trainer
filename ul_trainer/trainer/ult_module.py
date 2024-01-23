"""
    Abstract base class ul_trainer.ULTModule

"""

from abc import ABC
from abc import abstractmethod
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class UlModule(ABC, torch.nn.Module):
    """
        Abstract base class for any model that can be trained with our Trainer clas
        Create you classes by inheriting this class
            - methods decorated as @abstractmethod must be implemented in you class
                - [training_step, validation_step, train_dataloader, configure_optimizers]
            - other methods should be implemented but they are optional

    """
    @abstractmethod
    def training_step(self, batch, batch_idx=0) -> [torch.autograd.Variable, dict]:
        """
            Returns:
                loss: on which we can call .backward()
                logs: dicts with names as keys and some scalars as values
        """
        raise NotImplementedError

    @torch.no_grad()
    def validation_step(self, batch, batch_idx=0) -> dict:
        """
            Returns a dict with keys scalar and image:
                scalar: maps to a dictionary with scalar logs
                image: maps to a dictionary with image as values logs
        """
        # return {"scalar": scalar_logs, "image": image_logs}
        raise NotImplementedError

    @abstractmethod
    def train_dataloader(self) -> [DataLoader, DistributedSampler]:
        # return val_loader, val_sampler
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        # return self.optimizers, self.schedulers
        raise NotImplementedError

    def test_dataloader(self) -> [DataLoader, DistributedSampler]:
        # return val_loader, val_sampler
        raise NotImplementedError

    def val_dataloader(self) -> [DataLoader, DistributedSampler]:
        # return val_loader, val_sampler
        raise NotImplementedError
