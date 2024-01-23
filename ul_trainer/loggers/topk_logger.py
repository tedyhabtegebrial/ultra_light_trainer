import os
import math
import torch
import random
from ul_trainer.utils import create_colored_logger
from ul_trainer.utils import unlink_file
from ul_trainer.utils import exists
from ul_trainer.utils import make_dirs
from ul_trainer.utils import list_dir
from ul_trainer.utils import torch_load, torch_save


from ul_trainer.utils import save_image
from ul_trainer.utils import make_dirs
from ul_trainer.utils import torch_save


def arg_sort(scores, files, mode="max"):
    assert len(scores) == len(files), "score-file list mismatch"
    sorted_idxs = sorted(range(len(files)), key=scores.__getitem__)
    if mode == "max":
        sorted_idxs = list(reversed(sorted_idxs))
    scores = [scores[idx] for idx in sorted_idxs]
    files = [files[idx] for idx in sorted_idxs]
    return scores, files


class TopKLogger:
    def __init__(
        self,
        topk: int = 1,
        logdir: str = "temp",
        log_every_x_epoch: int = -1,
        log_every_x_step: int = -1,
        monitor: str = "none",
        monitor_mode: str = "max",
    ) -> None:
        self.rank_zero = int(os.environ["LOCAL_RANK"]) == 0
        self.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        self.logging_path = logdir
        # if self.rank_zero:
        self._safe_logdir(logdir)
        self.topk = topk
        self.log_every_x_epoch = log_every_x_epoch
        self.log_every_x_step = log_every_x_step
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.saved_model_paths = []
        self.saved_model_scores = []
        self.python_logger = create_colored_logger("TOP-K-Logger")

    def _safe_logdir(self, logdir):
        if self.rank_zero:
            make_dirs(logdir)
        return logdir

    def __repr__(self):
        repr_str = ""
        if len(self.saved_model_scores) < 0: return repr_str
        for score, path in zip(self.saved_model_scores,
                               self.saved_model_paths):
            repr_str += f"{path} ===== {round(score, 5)}"
            repr_str += "\n"
        return repr_str[:-1]

    def __str__(self):
        return self.__repr__()

    def state_dict(self):
        state_dict = {}
        state_dict["scores"] = self.saved_model_scores
        state_dict["paths"] = self.saved_model_paths
        return state_dict

    def get_latest(self, files):
        last_fname = os.path.join(self.logging_path, f"last.pt")
        if exists(last_fname):
            last_model = torch_load([last_fname], map_location=self.device)[0]
            if not isinstance(last_model, (dict, )):
                # when we have only saved a model object
                last_model = last_model.state_dict()
            epoch = last_model["EPOCH_IDX"]
            return last_fname, epoch
        epoch_list = []
        for f in files:
            f = os.path.basename(f)
            epoch = int(
                os.path.splitext(f)[0].split("epoch=")[-1].split("_")[0])
            epoch_list.append(epoch)
        epochs, files = arg_sort(epoch_list, files, "max")
        return files[0], epochs[0]

    def load(self, device=None):
        if device is None:
            device = self.device
        # if self.logging_path.startswith("gs://"):
        # self.logging_path = self.logging_path.replace("gs://", "/gcs/")
        # if len(files) < 1: return dict({})
        if list_dir(self.logging_path) in [None, []]:
            return None
        files = [f for f in list_dir(self.logging_path) if f.endswith(("pt", "t7"))]
        if len(files) < 1:
            return None
        files = [f for f in files if "score_" in f]
        if len(files) < 1:
            return None
        latest_model_name, epoch = self.get_latest(files)
        self.python_logger.debug(f"Loading from {latest_model_name} | RANK = {os.environ['LOCAL_RANK']}")
        ckpt_data = torch_load([latest_model_name], map_location=device)[0]
        k_0 = "MODEL_STATE_DICT"
        if not isinstance(ckpt_data, (dict, )):
            # we have only saved a model object
            ckpt_data = ckpt_data.state_dict()
            return {k_0: ckpt_data}
        # we need to return something that fits what is required in trainer
        if not ("EPOCH_IDX" in ckpt_data):
            ckpt_data["EPOCH_IDX"] = epoch
        if k_0 in ckpt_data:
            if not isinstance(ckpt_data[k_0], (dict,)):
                # we have saved meta data tha contains a model object
                ckpt_data[k_0] = ckpt_data[k_0].state_dict()
        return ckpt_data

    def restart(self):
        paths = sorted([f for f in list_dir(self.logging_path)])
        if len(paths)<1: return
        paths = [f for f in paths if f.endswith(("pt", "t7"))]
        paths = [f for f in paths if "score_" in f]
        if len(paths)<1: return
        # epochs = [int(os.path.splitext(f)[0].split("epoch=")[-1].split("_")[0]) for f in paths]
        paths = [os.path.join(self.logging_path, f) for f in paths]
        scores = [float(os.path.splitext(f)[0].split("score_")[-1]) for f in paths]
        if len(scores) < self.topk:
            # if limit not reached
            self.python_logger.info(f"Found {len(scores)} saved models")
            self.saved_model_paths = paths
            self.saved_model_scores = scores
        else:
            sorted_scores, sorted_files = arg_sort(scores,
                                                   paths,
                                                   mode=self.monitor_mode)
            sorted_scores = sorted_scores[:self.topk]
            sorted_files = sorted_files[:self.topk]
            self.saved_model_paths = sorted_files
            self.saved_model_scores = sorted_scores
            self.python_logger.info(f"Found {len(scores)} / {self.topk} saved models")

    def load_state_dict(self, state_dict):
        self.saved_model_scores = state_dict["scores"]
        self.saved_model_paths = state_dict["paths"]
        for f in self.saved_model_paths:
            assert exists(f), f"File not found {f}"
            log_base_dir = self.logging_path
            assert log_base_dir in f, f"File {f} should be in self.logging_path {log_base_dir}"

    def __call__(self, meta_data: dict = None):
        epoch = meta_data["EPOCH_IDX"]
        global_index = meta_data["GLOBAL_STEP"]
        score = meta_data["SCORE"]

        if not self.rank_zero:
            return
        if score is None:
            raise ValueError("you forgot to pass score to topk logger")
        msg = f"epoch={str(epoch).zfill(3)}" if epoch > - \
            1 else f"global_index={str(global_index).zfill(8)}"
        c_score = score[self.monitor]
        self._save_snapshot(
            meta_data, c_score, self.saved_model_scores,
            self.saved_model_paths, msg)

    def save(self, model, filename):
        torch_save(model, filename)

    def _save_snapshot(self, model, score, score_list, file_list, post_fix=""):
        last_fname = os.path.join(self.logging_path,f"last.pt")
        self.save(model, last_fname)
        fname = os.path.join(
            self.logging_path,
            f"ckpt_{post_fix}_{self.monitor.replace('/', '_')}_score_{str(round(score, 6))}.pt"
        )
        mode = self.monitor_mode
        topk = self.topk
        updated_scores = score_list + [score]
        updated_files = file_list + [fname]
        if len(file_list) < topk:
            # if limit not reached
            self.python_logger.debug(
                f"saving model with {self.monitor}={score} at {post_fix}")
            self.save(model, fname)  # add sate_dict()
            self.saved_model_scores, self.saved_model_paths = updated_scores, updated_files
            model["TOPK"] = self.state_dict()
        else:
            sorted_scores, sorted_files = arg_sort(updated_scores,
                                                   updated_files,
                                                   mode=mode)
            poped_file = sorted_files.pop()
            poped_score = sorted_scores.pop()

            if poped_file in file_list:
                # if the least accurate model is in out list
                self.python_logger.debug(
                    f"saving model with {self.monitor}={score} at {post_fix}")
                self.python_logger.debug(
                    f"deleting model with {self.monitor}={poped_score} at {post_fix}"
                )
                if exists(poped_file):
                    unlink_file(poped_file)
                self.saved_model_paths, self.saved_model_scores = sorted_files, sorted_scores
                model["TOPK"] = self.state_dict()
                self.save(model, fname)


if __name__ == '__main__':
    os.environ["LOCAL_RANK"] = "0"
    device = f"cuda:0"
    args = {}
    args["topk"] = 5
    args["logdir"] = "./temp_topk/exp_12"
    args["log_every_x_epoch"] = 10
    args["log_every_x_step"] = -1
    args["monitor"] = "mse"
    args["monitor_mode"] = "max"
    os.makedirs(args["logdir"], exist_ok=True)
    topk_saver = TopKLogger(**args)
    model = torch.nn.Sequential(torch.nn.Linear(10, 1)).to(device)
    import tqdm
    all_scores = []
    num_epochs = 100
    for epoch in tqdm.tqdm(range(num_epochs), total=num_epochs):
        for itr in range(1):
            score = random.random()
            if epoch % args["log_every_x_epoch"] == 0:
                all_scores.append(score)
            meta_data = {}
            meta_data["SCORE"] = {"mse": score}
            meta_data["EPOCH_IDX"] = epoch
            meta_data["GLOBAL_STEP"] = epoch + itr
            meta_data["MODEL_STATE_DICT"] = model.cpu().state_dict()
            meta_data["TOPK"] = topk_saver.state_dict()
            topk_saver(meta_data)
    all_scores = torch.Tensor(all_scores)
    top_k, _ = all_scores.sort()
    print(top_k[:args["topk"]])
    print(top_k[-args["topk"]:])
