import json
import os
import time

import torch

from src.utils.result_io import dataset_of

LOG_KEYS = (
    "model",
    "des",
    "seed",
    "data_path",
    "seq_len",
    "label_len",
    "pred_len",
    "enc_in",
    "d_model",
    "d_ff",
    "e_layers",
    "n_heads",
    "dropout",
    "learning_rate",
    "batch_size",
    "train_epochs",
    "patience",
    "loss",
    "dlt_variant",
    "dlt_k",
    "dlt_L",
    "dlt_R",
    "dlt_impl",
    "dlt_neighbours",
    "dlt_gate",
    "dlt_topk",
    "plug_base",
    "dlt_fusion",
    "dlt_train",
    "dlt_select",
    "dlt_lr",
)


def _root(args):
    return getattr(args, "checkpoints", "") or "./temp/checkpoints"


def checkpoint_dir(args):
    return os.path.join(_root(args), dataset_of(args))


def ckpt_prefix(args):
    des = getattr(args, "des", "")
    return "{}_{}_sl{}_pl{}{}_s{}".format(dataset_of(args), args.model, args.seq_len, args.pred_len, "_" + des if des else "", getattr(args, "seed", 0))


def checkpoint_path(args, seq):
    return os.path.join(checkpoint_dir(args), "{}_{}.pth".format(ckpt_prefix(args), seq))


def _log_path(args):
    return os.path.join(checkpoint_dir(args), "checkpoints_{}.jsonl".format(dataset_of(args)))


def next_seq(args):
    folder = checkpoint_dir(args)
    if not os.path.isdir(folder):
        return 1
    prefix = ckpt_prefix(args) + "_"
    seqs = [0]
    for f in os.listdir(folder):
        if f.endswith(".pth") and f.startswith(prefix):
            try:
                seqs.append(int(f[len(prefix):-4]))
            except ValueError:
                pass
    return max(seqs) + 1


def find_latest_checkpoint_seq(args):
    return next_seq(args) - 1


def checkpoint_exists(args, seq):
    return os.path.exists(checkpoint_path(args, seq))


def log_checkpoint(args, seq):
    """One JSON line per checkpoint with the run configuration, so that analyses can find checkpoints by configuration."""
    entry = {"file": os.path.basename(checkpoint_path(args, seq)), "seq": seq, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    entry.update({k: getattr(args, k) for k in LOG_KEYS if hasattr(args, k)})
    with open(_log_path(args), "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_checkpoint(args, model, seq, device="cpu"):
    path = checkpoint_path(args, seq)
    model.load_state_dict(torch.load(path, map_location=device))
    print("Loaded checkpoint {}".format(path))
    return path


def cleanup_checkpoint(args, seq):
    path = checkpoint_path(args, seq)
    if os.path.exists(path):
        os.remove(path)
        print("Removed checkpoint {}".format(path))


def find_checkpoints(root, dataset, **config):
    """Checkpoint files of a dataset whose logged configuration matches config (newest last)."""
    log = os.path.join(root, dataset, "checkpoints_{}.jsonl".format(dataset))
    out = []
    if not os.path.exists(log):
        return out
    for line in open(log):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if all(entry.get(k) == v for k, v in config.items()):
            path = os.path.join(root, dataset, entry["file"])
            if os.path.exists(path):
                out.append(path)
    return out
