import os
import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image as tvsave
from torch.utils.tensorboard import SummaryWriter
from ul_trainer.utils import save_image
from ul_trainer.utils import make_dirs
from ul_trainer.utils import torch_save




class TBLogger:
    def __init__(self, logdir: str, not_tb_logging: bool = False) -> None:
        self.rank_zero = int(os.environ["LOCAL_RANK"]) == 0
        self.not_tb_logging = not_tb_logging
        self.log_steps = 0
        if self.rank_zero:
            self.setup(logdir)

    def save_image_summary(self, image_dict: dict = None):
        _keys = list(image_dict.keys())
        for _k in _keys:
            print(_k)
        for k, v in image_dict.items():
            print(k, v.shape)
        num_gpus, batch_size = image_dict[_keys[0]].shape[:2]
        # num_gpus, batch_size, _c, height, width = image_dict[_keys[0]].shape
        # image_dict = {k:v.view(-1, c_, height, width) for k,v in image_dict.items()}
        # batch_size = image_dict[_keys[0]].shape[0]
        # exit()
        for p in range(num_gpus):
            # print(f"GPU-{p}")
            for itr in range(batch_size):
                # print(f"Batch-{itr}")
                for k, v in image_dict.items():
                    # print(k, v.shape)
                    _img = (v[p][itr].to("cpu").squeeze())
                    # print(k, _img.shape)
                    fname = f"step_{str(self.log_steps).zfill(8)}_{str(k).replace('/', '_')}.png"
                    fpath = os.path.join(self.imgs_path, fname)
                    # print(fpath)
                    save_image(fp=fpath, tensor=make_grid(_img))
                self.log_steps += 1

    def setup(self, logdir):
        self.logdir = self._safe_logdir(logdir)
        self.tb_logger = self._tb_logger()
        if self.rank_zero:
            self.imgs_path = os.path.join(self.logdir, "images")
            os.makedirs(self.imgs_path, exist_ok=True)

    def __call__(self, log_dict, global_index=0, dtype="scalar"):
        if not self.rank_zero:
            return
        if self.not_tb_logging:
            return
        if dtype == "scalar":

            def mean(x):
                return x.mean() if isinstance(x, (torch.Tensor, )) else x

            def item(x):
                return x.item() if hasattr(x, 'item') else x

            for k, v in log_dict.items():
                self.tb_logger.add_scalar(k,
                                          mean(item(v)),
                                          global_step=global_index)
                # self.tb_logger.flush()
        elif dtype == "image":
            log_dict = {k: v.to('cpu') for k, v in log_dict.items()}
            for k, v in log_dict.items():
                if len(v.shape) > 4:
                    v = v.view(-1, *v.shape[-3:])
                v = make_grid(v)
                self.tb_logger.add_image(k, v, global_step=global_index)
                # self.tb_logger.flush()
        else:
            raise NotImplementedError(f"unknown dtype={dtype}")

    def _tb_logger(self):
        if self.rank_zero and (not self.not_tb_logging):
            return SummaryWriter(self.logdir)
        else:
            return None

    def _safe_logdir(self, logdir):
        if not self.rank_zero:
            return logdir
        make_dirs(logdir, exist_ok=True)
        return logdir


if __name__ == '__main__':
    import random
    import tqdm
    os.environ["LOCAL_RANK"] = "0"
    logger = TBLogger(
        logdir="gs://giotto_private/diffusion_3d/logs/temp/tb_logger-3")
    scalar = {"rand_1": random.random(), "rand_2": random.random()}
    image = {
        "col_0": torch.rand(1, 3, 32, 32),
        "col_1": torch.rand(3, 32, 32),
        "col_2": torch.rand(2, 3, 32, 32),
        "gray_0": torch.rand(1, 1, 32, 32),
        "gray_1": torch.rand(1, 32, 32),
        "gray_2": torch.rand(2, 1, 32, 32),
        "gray_3": torch.rand(32, 32),
    }
    for itr in tqdm.tqdm(range(100), total=100):
        # print(itr)
        logger(scalar, global_index=itr, dtype="scalar")
        # logger(image, global_index=itr, dtype="image")
