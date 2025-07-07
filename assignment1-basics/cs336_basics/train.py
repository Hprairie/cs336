import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from argparse import Namespace

from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.optimizers import AdamW, SGD
from cs336_basics.lr_schedule import CosineAnnealing
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.losses import cross_entropy_loss
from cs336_basics.gradient_clipping import clip_gradients
from cs336_basics.data_loader import DataLoader, RandomSampler, OrderedSampler
from cs336_basics.logger import WandbLogger

from cs336_basics.utils.parser import ParseKVAction

SAMPLE_SIZE = 100_000


def tokenize_and_cache(cfg, tokenizer: Tokenizer, path: os.PathLike, logger: WandbLogger):
    name = os.path.basename(path).split(".")[0]  # Get filename
    # Check if data is cached -> if so load from disk
    cache_path = os.path.join(cfg.cache_path, f"{name}.npy")
    if os.path.exists(cache_path):
        logger.log(f"Loading from cache file {cache_path}")
        return np.memmap(cache_path, dtype=np.int16, mode="r")

    # Determine compression ratio to calculate expected number of bytes to allocate
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        sample = f.read(SAMPLE_SIZE)
    input_ids = tokenizer.encode(sample.decode("UTF-8", errors="replace"))
    ratio = len(sample) / len(input_ids)
    # Allocate memory for tensors
    data = np.memmap(cache_path, dtype=np.uint16, mode="w+", shape=(int(file_size / ratio),))
    # Tokenize that dataset
    idx = 0
    with open(path) as f:
        for ids in tqdm(tokenizer.encode_iterable(f)):
            data[idx] = ids
            idx += 1
    data.flush()
    data._mmap.close()
    # Truncate the file to the real size (2 bytes per token)
    with open(cache_path, "r+b") as f:
        f.truncate(idx * 2)
    return np.memmap(cache_path, dtype=np.int16, mode="r")


def get_lr_schedule(cfg: Namespace, optimizer: torch.optim.Optimizer):
    match cfg.lr_schedule:
        case "cosine":
            return CosineAnnealing(
                optimizer=optimizer,
                a_max=cfg.lr_max,
                a_min=cfg.lr_min,
                T_w=cfg.lr_schedule_vals["warmup"],
                T_c=cfg.train_steps,
            )
        case _:
            raise ValueError(f"Unknown LR Schedule: {cfg.lr_schedule}")


def get_optimizer(cfg: Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    match cfg.optimizer:
        case "adamw":
            return AdamW(model, cfg.max_lr)
        case _:
            raise ValueError(f"Unknown Optimizer: {cfg.optimizer}")


def validation(
    cfg,
    model: TransformerLM,
    data: DataLoader,
    criterion,
    logger: WandbLogger,
    iteration: int,
    device: torch.device,
):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        samples = 0
        for x, y in data:
            outputs = model(x)
            loss: torch.Tensor = criterion(outputs, y)
            total_loss += loss.detach().cpu().item()
            samples += 1
        logger.log(
            data={
                "val/loss": total_loss / samples,
            },
            step=iteration,
        )


def train(cfg: Namespace) -> None:
    assert not os.path.exists(
        os.path.join(cfg.run_dir, cfg.run_name)
    ), "Experiment all ready exists. Please change run name."
    # Setup device, logger, and dump info
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = WandbLogger(cfg)
    # Setup of model/tokenizer/optimizer/etc.
    logger.log("Creating Tokenizer and Model...")
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

    logger.log("Building Criterion, Optimizer, and LR Schedule...")
    criterion = cross_entropy_loss
    optimizer = get_optimizer(cfg=cfg, model=model)
    lr_schedule = get_lr_schedule(cfg=cfg, optimizer=optimizer)
    logger.log(f"Using Optimizer: {type(optimizer)}")
    logger.log(f"Using Criterion: {criterion.__name__}")
    logger.log(f"Using LR Schedule: {type(lr_schedule)}")

    logger.log("Tokenizing Data...")
    train_data = tokenize_and_cache(cfg=cfg, tokenizer=tokenizer, path=cfg.train_data_path, logger=logger)
    val_data = tokenize_and_cache(cfg=cfg, tokenizer=tokenizer, path=cfg.train_data_path, logger=logger)

    logger.log("Creating Run Directory...")
    save_path = os.path.join(cfg.run_dir, cfg.run_name, "checkpoints")
    os.makedirs(save_path, exist_ok=True)

    # Create the dataloaders
    logger.log("Building Dataloaders...")
    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(
            context_length=cfg.context_length,
            dataset_size=len(train_data),
        ),
        steps=cfg.train_steps,
        batch_size=cfg.batch_size,
        context_length=cfg.contex_length,
        pin_memory=True,
        drop_last=False,  # This Does nothing rn
        device=device,
        dtype=torch.bfloat16,
    )

    val_dataloader = DataLoader(
        val_data,
        sampler=OrderedSampler(
            context_length=cfg.context_length,
            dataset_size=len(val_data),
        ),
        batch_size=cfg.batch_size,
        context_length=cfg.contex_length,
        pin_memory=True,
        drop_last=False,  # This Does nothing rn
        device=device,
        dtype=torch.bfloat16,
    )

    # Main training loop
    for step, (x, y) in enumerate(train_dataloader):
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        lr_schedule.step(t=step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.eval_steps == 0:
            validation(
                cfg=cfg,
                model=model,
                data=val_dataloader,
                criterion=criterion,
                logger=logger,
                iteration=step,
                device=device,
            )

        if step % cfg.save_steps == 0:
            save_checkpoint(
                model=model, optimizer=optimizer, iteration=step, out=os.path.join(save_path, f"checkpoint_{step}.pth")
            )

        if step % cfg.log_steps == 0:
            logger.log(
                data={
                    "train/loss": loss.detach().cpu().item(),
                    "train/learning-rate": optimizer.param_groups[0]["lr"],
                },
                step=step,
            )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=os.PathLike, default="runs/")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--tokenizer", type=os.PathLike, default="tokenizer/")

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=25)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("-rope-theta", type=int, default=10000)

    parser.add_argument("--optimizer", type="str", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument(
        "--optimizer-vals",
        nargs="+",
        action=ParseKVAction,
        default={
            "betas": (0.9, 0.999),
            "weight_decay": 0,
        },
    )
    parser.add_argument("--max-lr", type=int, default=1e-3)
    parser.add_argument("--min-lr", type=int, default=1e-5)
    parser.add_argument("--steps", type=int, default=10_000)

    parser.add_argument("--lr-schedule", type="str", choices=["lr_cosine"], default="lr_cosine")
    parser.add_argument(
        "--lr-schedule-vals",
        nargs="+",
        action=ParseKVAction,
        default={
            "warmup": 100,
        },
    )

    parser.add_argument("--train-data-path", type=os.PathLike, default="data/")
    parser.add_argument("--val-data-path", type=os.PathLike, default="data/")
    parser.add_argument("--cache-path", type=os.PathLike, default="tokenized_data/")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=2_000)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)

    parser.add_argument("--wandb-entity", type="str")
    parser.add_argument("--wandb-project", type="str")
    parser.add_argument("--wandb-name", type="str")

    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
