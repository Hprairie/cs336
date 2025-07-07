import math
import argparse
import timeit
import torch
import numpy as np

from argparse import Namespace
from torch.cuda import nvtx
from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum

import cs336_basics

from cs336_basics.model import BasicsTransformerLM, softmax
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

configs = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "vocab_size": 10_000,
        "context_length": 256,
        "rope_theta": 10_000,
    },
}


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(
            attention_scores, dim=-1
        )  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        final = einsum(
            attention_weights, V, "... query key, ... key d_v ->  ... query d_v"
        )

    return final


cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def get_random_batch(cfg: Namespace) -> torch.Tensor:
    return torch.randint(
        0, configs[cfg.model]["vocab_size"], (cfg.batch_size, cfg.context_window)
    )


@nvtx.range("Forward Model")
def forward(
    cfg: Namespace, model: BasicsTransformerLM, device: torch.device
) -> tuple[float, float]:
    for _ in range(cfg.warmup_steps):
        data = get_random_batch(cfg=cfg).to(device=device)
        _ = model(data)
        torch.cuda.synchronize()

    times = []
    for _ in range(cfg.benchmark_steps):
        data = get_random_batch(cfg=cfg).to(device=device)
        start_time = timeit.default_timer()
        with nvtx.range("Forward-Pass"):
            _ = model(data)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    return np.average(times).item(), np.std(times).item()


@nvtx.range("Forward-Backward Model")
def forward_backward(
    cfg: Namespace, model: BasicsTransformerLM, device: torch.device
) -> tuple[float, float]:
    criterion = cross_entropy
    for _ in range(cfg.warmup_steps):
        x = get_random_batch(cfg=cfg).to(device=device)
        y = get_random_batch(cfg=cfg).to(device=device)
        logits = model(x)
        loss: torch.Tensor = criterion(logits, y)
        loss.backward()
        torch.cuda.synchronize()

    times = []
    for _ in range(cfg.benchmark_steps):
        x = get_random_batch(cfg=cfg).to(device=device)
        y = get_random_batch(cfg=cfg).to(device=device)
        start_time = timeit.default_timer()
        with nvtx.range("Forward-Pass"):
            logits = model(x)
            loss: torch.Tensor = criterion(logits, y)
        with nvtx.range("Backward-Pass"):
            loss.backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    return np.average(times).item(), np.std(times).item()


@nvtx.range("Forward-Backward-Optimizer Model")
def forward_backward_optimizer(
    cfg: Namespace, model: BasicsTransformerLM, device: torch.device
) -> tuple[float, float]:
    criterion = cross_entropy
    optimizer = AdamW(model.parameters())
    for _ in range(cfg.warmup_steps):
        optimizer.zero()
        x = get_random_batch(cfg=cfg).to(device=device)
        y = get_random_batch(cfg=cfg).to(device=device)
        logits = model(x)
        loss: torch.Tensor = criterion(logits, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    times = []
    for _ in range(cfg.benchmark_steps):
        x = get_random_batch(cfg=cfg).to(device=device)
        y = get_random_batch(cfg=cfg).to(device=device)
        start_time = timeit.default_timer()
        optimizer.zero()
        with nvtx.range("Forward-Pass"):
            logits = model(x)
            loss: torch.Tensor = criterion(logits, y)
        with nvtx.range("Backward-Pass"):
            loss.backward()
        with nvtx.range("Optimizer-Pass"):
            optimizer.step()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    return np.average(times).item(), np.std(times).item()


def benchmark(cfg: Namespace) -> None:
    model_config = configs[cfg.model]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = BasicsTransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
    )

    model = model.to(device)

    match cfg.type:
        case "forward":
            avg_time, std = forward(cfg=cfg, model=model, device=device)
            print(f"Average Forward computation time: {avg_time}")
            print(f"Std Forward computation time: {std}")
        case "forward-backward":
            avg_time, std = forward_backward(cfg=cfg, model=model, device=device)
            print(f"Average Forward and Backward computation time: {avg_time}")
            print(f"Std Forward and Backward computation time: {std}")
        case "forward-backward-optimizer":
            avg_time, std = forward_backward_optimizer(
                cfg=cfg, model=model, device=device
            )
            print(f"Average Forward and Backward computation time: {avg_time}")
            print(f"Std Forward and Backward computation time: {std}")
        case _:
            raise ValueError(f"Unknown Benchmarking type {cfg.type}")


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser("Benchmark Model Script")
    parser.add_argument("--model", type=str, choices=configs.keys(), default="small")
    parser.add_argument(
        "--type",
        type=str,
        choices=["forward", "forward-backward", "forward-backward-optimizer"],
        default="forward",
    )

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=10)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-window", type=int, default=256)

    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    benchmark(cfg=cfg)
