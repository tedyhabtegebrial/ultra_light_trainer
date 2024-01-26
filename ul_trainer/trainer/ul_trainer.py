import os
import tqdm
import torch
import random
import numpy as np
import gc
import tempfile
import json
from contextlib import contextmanager
import torch.distributed as dist
import tempfile

from ul_trainer.loggers import TBLogger
from ul_trainer.loggers import TopKLogger

from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from hydra import initialize, compose

from ul_trainer.utils import create_colored_logger
from ul_trainer.utils import choice, dummy_progress_bar
from ul_trainer.utils import torch_save
from ul_trainer.utils import torch_load
from ul_trainer.utils import exists, list_dir
from ul_trainer.ema import EMA
import torch.nn.functional as F

import torch.nn as nn


def seed_all(seed=43):
    return
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)

def ddp_setup():
    init_process_group(backend="nccl")

def ddp_destroy():
    destroy_process_group()

def save_hparams(content, filename):
    OmegaConf.save(config=content, f=filename)


class UlTrainer:
    def __init__(
        self,
        model,
        tb_logger: TBLogger = None,
        topk_logger: TopKLogger = None,
        validate_every_x_epoch: int = -1,
        max_epochs: int = 100,
        log_root: str = None,
        tb_log_steps: int = 10,
        use_ema: bool = False,
        ema_update_interval: int = 5,
    ) -> None:
        self._set_device_and_rank()
        self.logdir = log_root
        self.configs_save = True
        self.cmd_logger = create_colored_logger("TRAINER")
        self.tb_log_steps = tb_log_steps
        self.global_step = 0
        self.epoch_idx = 0
        self.max_epochs = max_epochs
        self.validate_every_x_epoch = validate_every_x_epoch
        # Loading weights for the model
        self.model = model.to(self.device)
        self.tb_logger = tb_logger
        self.use_ema = use_ema
        self.ema_update_interval = ema_update_interval
        self.validation_global_idx = 0
        self.validation_scalar_idx = 0
        self.validation_image_idx = 0
        self.topk_logger = topk_logger
        self._configure_optimizers()
        self._setup_ema()
        ddp_setup()
        self.resume()
        if is_initialized():
            self.model = DDP(self.model.to(self.device), device_ids=[int(os.environ["LOCAL_RANK"])])
        else:
            raise RuntimeError("No Distributed Training Initialized")
        if self.rank_zero and self.configs_save is not None:
            os.makedirs(self.logdir, exist_ok=True)

    def _setup_ema(self):
        self.model_ema = None
        if self.use_ema:
            self.model_ema = EMA(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            model_ = self.model.module if is_initialized() else self.model
            self.model_ema.store(model_)
            self.model_ema.copy_to(model_)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                model_ = self.model.module if is_initialized() else self.model
                self.model_ema.restore(model_)
                if context is not None:
                    print(f"{context}: Restored training weights")

    def _set_device_and_rank(self):
        if not torch.cuda.is_available():
            self.device = "cpu"
            if "LOCAL_RANK" in os.environ:
                self.rank_zero = int(os.environ['LOCAL_RANK']) == 0
            else:
                self.rank_zero = True
        else:
            if "LOCAL_RANK" in os.environ:
                torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
                self.device = f"cuda:{os.environ['LOCAL_RANK']}"
                self.rank_zero = int(os.environ['LOCAL_RANK']) == 0
            else:
                torch.cuda.set_device(0)
                self.device = "cuda:0"
                self.rank_zero = True

    def load_ckpt(self, ckpt_path):
        if self.rank_zero:
            self.cmd_logger.info("Loading ckpt ")
        assert exists(ckpt_path), "ckpt file not found"
        all_state_dict = torch_load(ckpt_path, map_location=self.device)
        state_dict = all_state_dict["MODEL_STATE_DICT"]
        self.model.module.load_state_dict(state_dict)

    def resume(self):
        """
            Resumes training when there is already some data saved
        """
        if self.topk_logger is None:
            self.cmd_logger.info("No topk found | Fresh training")
            return
        topk_path = self.topk_logger.logging_path
        if list_dir(topk_path) in [None, []]:
            self.cmd_logger.info("No Model Found in logging dir")
            return
        self.cmd_logger.debug(f"Loading snapshot from {topk_path}")
        model_snapshot = self.topk_logger.load(device=self.device)
        if model_snapshot is None:
            return
        if "MODEL_STATE_DICT" in model_snapshot:
            self.model.load_state_dict(model_snapshot["MODEL_STATE_DICT"])
            self.cmd_logger.debug(f"Saved Snaphot is found at {topk_path}")
        else:
            self.cmd_logger.info(f"No Save Model found at {topk_path}")
            model_snapshot = {}

        if "EMA_MODEL_STATE_DICT" in model_snapshot:
            ema_sd = model_snapshot["EMA_MODEL_STATE_DICT"]
            if self.use_ema:
                self.model_ema.load_state_dict(ema_sd)
                self.cmd_logger.info("FOUND Saved EMA Parameters")

        self.topk_logger.restart()
        if "EPOCH_IDX" in model_snapshot:
            self.epoch_idx = model_snapshot["EPOCH_IDX"] # + 1
        if "GLOBAL_STEP" in model_snapshot:
            self.global_step = model_snapshot["GLOBAL_STEP"]
        else:
            dataset_len = len(self.model.train_dataloader())
            self.global_step = dataset_len*self.epoch_idx
        for i, sch in enumerate(self.scheduler_list):
            if f"SCHEDULER_{i}_STATE_DICT" in model_snapshot:
                if sch is not None:
                    sch.load_state_dict(
                        model_snapshot[f"SCHEDULER_{i}_STATE_DICT"])
        self.cmd_logger.info(f"Rank == {os.environ['LOCAL_RANK']} loading snapshot FINISHED")
        self.cmd_logger.info(f"Rank == {os.environ['LOCAL_RANK']} Found {model_snapshot.keys()}")

    def batch_process(self, batch, mode='train'):
        return self.copy_to_gpu(batch)

    def copy_to_gpu(self, batch):
        if isinstance(batch, (dict,)):
            batch = {k: self.copy_to_gpu(v) for k, v in batch.items()}
            return batch
        elif isinstance(batch, (list,)):
            batch = [self.copy_to_gpu(b) for b in batch]
            return batch
        elif isinstance(batch, (torch.Tensor)):
            batch = batch.to(self.device)
            return batch
        elif isinstance(batch, (str)):
            return batch
        else:
            return batch

    def _configure_optimizers(self):
        assert hasattr(self, 'model'), "model not defined"
        assert hasattr(self.model, 'configure_optimizers')
        optimizer_list, scheduler_list = self.model.configure_optimizers()
        self.optimizer_list = optimizer_list
        self.scheduler_list = scheduler_list
        if hasattr(self.model, "weight_load_summary"):
            if self.rank_zero:
                with open(os.path.join(self.logdir, "w_load_summary.json"), "w") as fid:
                    json.dump(fp=fid, obj=self.model.weight_load_summary, indent=4)

    def _clip_grads(self):
        # # clip_grads
        # for opt in self.optimizer_list:
        #     if opt is not None:
        #         torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 100)
        pass

    def _optimizer_step(self):
        for opt in self.optimizer_list:
            if opt is not None:
                opt.step()

    def _scheduler_step(self):
        for sch in self.scheduler_list:
            if sch is not None:
                sch.step()

    def _optimizer_zero_grad(self):
        for opt in self.optimizer_list:
            if opt is not None:
                opt.zero_grad()

    def on_epoch_start(self):
        return

    def _get_topk_meta_data(self, model, epoch, global_idx, score):
        meta_data = {}
        meta_data["SCORE"] = score
        meta_data["EPOCH_IDX"] = epoch
        meta_data["GLOBAL_STEP"] = global_idx
        meta_data["MODEL_STATE_DICT"] = model.cpu().state_dict()
        meta_data["TOPK"] = self.topk_logger.state_dict()
        if self.use_ema:
            meta_data["EMA_MODEL_STATE_DICT"] = self.model_ema.state_dict()
        model.to(self.device)
        for i, optim in enumerate(self.optimizer_list):
            if optim is not None:
                meta_data[f"OPTIMIZER_{i}_STATE_DICT"] = optim.state_dict()
        for i, sch in enumerate(self.scheduler_list):
            if sch is not None:
                meta_data[f"SCHEDULER_{i}_STATE_DICT"] = sch.state_dict()
        return meta_data

    def train(self, train_data: DataLoader = None, train_sampler: DistributedSampler = None):
        train_data, train_sampler = choice(
            self.model.module.train_dataloader(),
            [train_data, train_sampler],
            train_data is None)
        for epoch in range(self.epoch_idx, self.max_epochs):
            # if epoch==0:
            #     t_file = tempfile.NamedTemporaryFile(dir="./")
            self.model.module.eval()
            val_summary = self.validate()
            self.model.module.train()
            if self.validate_every_x_epoch > 0 and (self.epoch_idx % self.validate_every_x_epoch == 0):
                if self.rank_zero:
                    self.cmd_logger.info(f"Starting Validation Epoch {epoch}")
                self.model.module.eval()
                with torch.no_grad():
                    val_summary = self.validate(using_ema=False)
                    torch.cuda.empty_cache()
                    if self.use_ema:
                        with self.ema_scope("validate: "):
                            val_summary_ema = self.validate(using_ema=True)
                            val_summary.update(val_summary_ema)
                self.topk_logger(
                    self._get_topk_meta_data(self.model.module,
                                             self.epoch_idx,
                                             self.global_step,
                                             score=val_summary))
                self.model.module.train()
            barrier()
                
            if self.rank_zero:
                self.cmd_logger.info(f"Starting Training Epoch {epoch}")
            self.on_epoch_start()  # TODO
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self._optimizer_zero_grad()
            with tqdm.tqdm(total=len(train_data)
                           ) if self.rank_zero else dummy_progress_bar() as pb:
                for batch_idx, batch in enumerate(train_data):
                    batch = self.batch_process(batch)
                    pb_logs = self._train_step(batch, batch_idx)
                    if self.global_step % self.tb_log_steps == 0:
                        if self.tb_logger is not None:
                            self.tb_logger(pb_logs,
                                           self.global_step,
                                           dtype="scalar")
                    if self.rank_zero:
                        pb_logs = {k: f'{v:.3f}' for k, v in pb_logs.items()}
                        pb_logs.update({'epoch': epoch})
                        pb_logs.update({'step': self.global_step})
                        pb.set_postfix(**pb_logs)
                        pb.update(1)
                    self.global_step += 1
                    if batch_idx % 10 == 0:
                        gc.collect()
            self.epoch_idx += 1
            torch.cuda.empty_cache()
        # self.model.module.to("cpu")
        # torch_save(self.model.module.state_dict(), self.final_model_path)
        # self.model.module.to(self.device)
        destroy_process_group()

    def train_gan(self, train_data: DataLoader = None, train_sampler: DistributedSampler = None):
        raise NotImplementedError

    @torch.no_grad()
    def validate(self, val_data=None, using_ema=False):
        val_summary = {}
        val_data = choice(self.model.module.val_dataloader()
                          [0], val_data, val_data is None)
        assert len(val_data) > 0, "validation data length ==0"
        with tqdm.tqdm(total=len(val_data)) if self.rank_zero else dummy_progress_bar() as pb:
            if self.rank_zero:
                self.cmd_logger.info(
                    f"Starting Validation at Epoch  {self.epoch_idx}")
            for batch_idx, batch in enumerate(val_data):
                batch = self.batch_process(batch)
                logs = self.model.module.validation_step(batch, batch_idx)
                if using_ema:
                    logs["scalar"] = {
                        k + "_ema": v
                        for k, v in logs["scalar"].items()
                    }
                    logs["image"] = {
                        k + "_ema": v
                        for k, v in logs["image"].items()
                    }
                scalar_logs, img_logs = logs["scalar"], logs["image"]
                scalar_logs = self._gather_logs(scalar_logs, dtype="scalar")
                if batch_idx == 0:
                    val_summary = scalar_logs
                    val_summary = {k: [v] for k, v in scalar_logs.items()}
                else:
                    for k, v in scalar_logs.items():
                        val_summary[k].append(v)
                self.validation_scalar_idx += 1
                if batch_idx == 0 and len(img_logs)>0:
                    img_logs = self._gather_logs(img_logs, dtype="image")
                    if self.tb_logger is not None:
                        self.tb_logger(
                            img_logs, self.validation_image_idx, dtype="image")
                    self.validation_image_idx += 1
                if self.rank_zero:
                    pb_logs = {k: f'{v:.3f}' for k, v in scalar_logs.items()}
                    pb.set_postfix(**pb_logs)
                    pb.update(1)
                    if batch_idx == 0 and len(img_logs) > 0:
                        if self.tb_logger is not None:
                            self.tb_logger.save_image_summary(img_logs)
        val_summary = {k: sum(v)/len(v) for k, v in val_summary.items()}
        if self.tb_logger is not None:
            self.tb_logger(val_summary, self.epoch_idx, dtype="scalar")
        return val_summary

    def _gather_logs(self, logs, dtype="scalar"):
        device = self.device
        if dtype == "scalar":
            logs = {k: torch.Tensor([v]).to(self.device)
                    for k, v in logs.items()}
            g_tensor = {k: torch.Tensor([v]).to(device) for k, v in logs.items()}
            for k in g_tensor:
                dist.reduce(g_tensor[k], 0, op=dist.ReduceOp.SUM)
                g_tensor[k] = g_tensor[k] / torch.cuda.device_count()
                g_tensor[k]  = g_tensor[k].item()
            gathered_logs = g_tensor
        else:
            logs = {k: v.to(self.device).contiguous() for k, v in logs.items()}
            gathered_logs = {k: [] for k, v in logs.items()}
            for k in logs.keys():
                for d in range(torch.cuda.device_count()):
                    gathered_logs[k].append(torch.ones_like(logs[k]).to(self.device))
                dist.all_gather(gathered_logs[k], logs[k])
            gathered_logs = {k: torch.stack(v).to("cpu")
                             for k, v in gathered_logs.items()}

        return gathered_logs

    def _train_step(self, batch, batch_idx=0):
        self._optimizer_zero_grad()
        loss, logs = self.model.module.training_step(batch, batch_idx)
        if loss is not None:
            loss.backward()
        logs_agg = self._gather_logs(logs, "scalar")
        self._clip_grads()
        self._optimizer_step()
        self._optimizer_zero_grad()
        self._scheduler_step()
        with torch.no_grad():
            if self.use_ema and ((self.global_step % self.ema_update_interval)==0):
                if is_initialized():
                    self.model_ema(self.model.module)
                else:
                    self.model_ema(self.model)
        return logs_agg

    def run_eval(self, val_data: DataLoader = None, val_sampler: DistributedSampler = None):
        self.model.module.eval()
        if self.tb_logger is not None:
            self.tb_logger.log_steps = 0
        with tqdm.tqdm(total=len(val_data)) if self.rank_zero else dummy_progress_bar() as pb:
            if self.rank_zero:
                self.cmd_logger.info(
                    f"Starting Validation at Epoch  {self.epoch_idx}")
            scalar_summary = {}
            for batch_idx, batch in enumerate(val_data):
                batch = self.batch_process(batch)
                logs = self.model.module.validation_step(batch, batch_idx=0)
                if self.use_ema:
                    with self.ema_scope("Run Eval"):
                        logs_ema = self.model.module.validation_step(batch, batch_idx)
                    logs_ema["scalar"] = {
                        k + "_ema": v
                        for k, v in logs_ema["scalar"].items()
                    }
                    logs_ema["image"] = {
                        k + "_ema": v
                        for k, v in logs_ema["image"].items()
                    }
                    logs["scalar"].update(logs_ema["scalar"])
                    logs["image"].update(logs_ema["image"])
                scalar_logs, img_logs = logs["scalar"], logs["image"]
                scalar_logs = self._gather_logs(scalar_logs, dtype="scalar")
                if batch_idx == 0:
                    for k, v in scalar_logs.items():
                        scalar_summary[k] = [float(v)]
                else:
                    for k, v in scalar_logs.items():
                        scalar_summary[k].append(float(v))
                img_logs = self._gather_logs(img_logs, dtype="image")
                self.validation_image_idx += 1
                if self.rank_zero:
                    pb_logs = {k: f'{v:.3f}' for k, v in scalar_logs.items()}
                    pb.set_postfix(**pb_logs)
                    pb.update(1)
                if self.rank_zero:
                    if self.tb_logger is not None:
                        self.tb_logger.save_image_summary(img_logs)
        self.model.module.train()
