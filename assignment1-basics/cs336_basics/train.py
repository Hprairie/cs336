import os
import torch
import argparse
import numpy as np


from argparse import Namespace

from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.optimizers import AdamW, SGD
from cs336_basics.lr_schedule import lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer 
from cs336_basics.losses import cross_entropy_loss
from cs336_basics.gradient_clipping import clip_gradients
from cs336_basics.data_loader import data_loader
from cs336_basics.logger import WandbLogger

from cs336_basics.utils.parser import ParseKVAction

SAMPLE_SIZE = 100_000

def tokenize_and_cache(cfg, tokenizer: Tokenizer, path: os.PathLike, logger: WandbLogger):
    name = os.path.basename(path).split('.')[0] # Get filename
    # Check if data is cached -> if so load from disk
    cache_path = os.path.join(cfg.cache_path, f"{name}.npy")
    if os.path.exists(cache_path):
        logger.log(f"Loading from cache file {cache_path}")
        return np.memmap(cache_path, dtype=np.int32)
    
    # Determine compression ratio to calculate expected number of bytes to allocate
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        sample = f.read(SAMPLE_SIZE)
    input_ids = tokenizer.encode(sample.decode("UTF-8", errors="replace"))
    ratio = len(sample) / len(input_ids)

    # Allocate memory for tensors
    data = np.memmap(
        cache_path,
        dtype=np.uint16,
        shape=(int(file_size / ratio),)
    )
    
    # Tokenize that dataset
    tokenized_text = []
    with open(path) as f:
        for ids in tokenizer.encode_iterable(f):
            tokenized_text.append(ids)
    
    # Truncate dataset to correct size
    tokenized_text = np.array(tokenized_text)
    data.flush()
    return data
    

def get_lr_schedule(cfg: Namespace):
    return

def get_optimizer(cfg: Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    match cfg.optimizer:
        case "adamw":
            return AdamW(model, cfg.lr)
        case _:
            raise ValueError(f"Unknown Optimizer: {cfg.optimizer}")
    

def train(cfg: Namespace) -> None:
    # Setup device, logger, and dump info
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = WandbLogger(cfg)
    # Setup of model/tokenizer/optimizer/etc.
    tokenizer = Tokenizer.from_files(
        os.path.join(cfg.tokenizer, "vocab.pkl"),
        os.path.join(cfg.tokenizer, "merges.pkl"),
        special_tokens=["<|endoftext|>"],
    )

    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=tokenizer.get_vocab_size(),
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        theta=cfg.rope_theta,
        device=device,
        dtype=torch.float32,
    )

    criterion = cross_entropy_loss
    lr_schedule = get_lr_schedule(cfg=cfg)
    optimizer = get_optimizer(cfg=cfg, model=model)

    data = ...

    # Main training loop
    for step in range(cfg.step_count):
        x, y = data_loader(data, cfg.batch_size, cfg.context_length, device=device)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=os.PathLike, default="runs/")
    parser.add_argument("--tokenizer", type=os.PathLike, default="tokenizer/")

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=25)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("-rope-theta", type=int, default=10000)

    parser.add_argument("--optimizer", type="str", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--optimizer-vals", nargs='+', action=ParseKVAction, default={
        "betas": (0.9, 0.999),
        "weight_decay": 0,
    })
    parser.add_argument("--max-lr", type=int, default=1e-3)
    parser.add_argument("--min-lr", type=int, default=1e-5)
    parser.add_argument("--steps", type=int, default=10_000)

    parser.add_argument("--lr-schedule", type="str", choices=["lr_cosine"], default="lr_cosine")
    parser.add_argument("--lr-schedule-vals", nargs='+', action=ParseKVAction, default={
        "warmup": 100,
    })

    parser.add_argument("--train-data-path", type=os.PathLike, default="data/")
    parser.add_argument("--val-data-path", type=os.PathLike, default="data/")
    parser.add_argument("--cache-path", type=os.PathLike, default="tokenized_data/")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--step-count", type=int, default=2_000)

    parser.add_argument("--wandb-entity", type="str")
    parser.add_argument("--wandb-project", type="str")
    parser.add_argument("--wandb-name", type="str")

    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)