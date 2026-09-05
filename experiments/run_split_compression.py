"""One-shot data-dependent compression PAC--Bayes benchmark.

The source paper's selected data-dependent configurations all use intrinsic
dimension d=0.  Consequently the posterior equals the model learned on split A,
the update message is empty, and split B is used only for certification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from pactl.benchmark_data import make_benchmark_datasets
from pactl.bounds.get_pac_bounds import compute_catoni_bound, pac_bayes_bound_opt
from pactl.paper_models import create_paper_model


ARCHITECTURES = {
    "mnist": "two-block-cnn-standard-head",
    "cifar10": "preact-wrn-28-4",
    "cifar100": "preact-wrn-28-4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--prior-fraction", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Per-process batch size under torchrun.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--misc-extra-bits", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    parser.add_argument("--sync-bn", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-examples", type=int, default=None,
                        help="Smoke tests only; a run using this is marked non-reportable.")
    return parser.parse_args()


def initialize_runtime(requested_device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if requested_device not in ("auto", "cuda"):
            raise ValueError("torchrun requires CUDA for this benchmark")
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun requested but CUDA is unavailable")
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), rank, world_size

    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested_device), 0, 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset,
    indices: np.ndarray | None,
    batch_size: int,
    workers: int,
    device: torch.device,
    rank: int,
    world_size: int,
    shuffle: bool,
    seed: int,
):
    selected = dataset if indices is None else Subset(dataset, indices.tolist())
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            selected,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
    loader = DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    return loader


def reduce_sum(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return value


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    world_size: int,
) -> float:
    model.train()
    loss_sum = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * targets.numel()
        count += targets.numel()
    reduce_sum(loss_sum, world_size)
    reduce_sum(count, world_size)
    return (loss_sum / count).item()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
):
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.float64)
    nll = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == targets).sum().double()
        nll += nn.functional.cross_entropy(logits, targets, reduction="sum").double()
        count += targets.numel()
    reduce_sum(correct, world_size)
    reduce_sum(nll, world_size)
    reduce_sum(count, world_size)
    return {
        "count": int(count.item()),
        "accuracy": (correct / count).item(),
        "error": 1.0 - (correct / count).item(),
        "average_nll": (nll / count).item(),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.prior_fraction < 1.0:
        raise ValueError("--prior-fraction must be strictly between zero and one")
    if not 0.0 < args.delta < 1.0:
        raise ValueError("--delta must be strictly between zero and one")

    device, rank, world_size = initialize_runtime(args.device)
    seed_everything(args.seed)
    started = time.time()

    output_dir = Path(
        args.output_dir or f"runs/{args.dataset}-split-compression-seed{args.seed}"
    ).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_aug, train_eval, test_eval = make_benchmark_datasets(
        args.dataset, args.data_root
    )
    available = len(train_aug)
    total = available
    reportable = True
    if args.max_train_examples is not None:
        if not 2 <= args.max_train_examples <= available:
            raise ValueError("--max-train-examples must be between 2 and dataset size")
        total = args.max_train_examples
        reportable = False

    permutation = np.random.default_rng(args.seed).permutation(available)[:total]
    split_at = int(round(args.prior_fraction * total))
    indices_a = permutation[:split_at]
    indices_b = permutation[split_at:]
    if not len(indices_a) or not len(indices_b):
        raise ValueError("The selected split produced an empty subset")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        workers=args.num_workers,
        device=device,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    train_a = make_loader(train_aug, indices_a, shuffle=True, **loader_kwargs)
    eval_a = make_loader(train_eval, indices_a, shuffle=False, **loader_kwargs)
    certify_b = make_loader(train_eval, indices_b, shuffle=False, **loader_kwargs)
    test = make_loader(test_eval, None, shuffle=False, **loader_kwargs)

    model = create_paper_model(args.dataset).to(device)
    if world_size > 1 and args.sync_bn and args.dataset.startswith("cifar"):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index], broadcast_buffers=True
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )
    if rank == 0:
        print(
            f"dataset={args.dataset} architecture={ARCHITECTURES[args.dataset]} "
            f"device={device} world_size={world_size} n_A={len(indices_a)} "
            f"n_B={len(indices_b)}"
        )
    for epoch in range(args.epochs):
        if isinstance(train_a.sampler, DistributedSampler):
            train_a.sampler.set_epoch(epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        average_loss = train_epoch(model, train_a, optimizer, device, world_size)
        if rank == 0 and (
            epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.epochs
        ):
            print(
                f"epoch={epoch + 1}/{args.epochs} train_A_nll={average_loss:.6f} "
                f"lr={current_lr:.8f}"
            )
        scheduler.step()

    metrics_a = evaluate(model, eval_a, device, world_size)
    metrics_b = evaluate(model, certify_b, device, world_size)
    metrics_test = evaluate(model, test, device, world_size)

    # The source paper reports d=0 for all selected data-dependent bounds.
    intrinsic_dim = 0
    message_bits = 0.0
    divergence_nats = (message_bits + args.misc_extra_bits) * math.log(2.0)
    if metrics_b["accuracy"] >= 0.5:
        certificate = pac_bayes_bound_opt(
            divergence=divergence_nats,
            train_error=metrics_b["error"],
            n=metrics_b["count"],
            epsilon=args.delta,
        )
        bound_name = "optimized_catoni"
    else:
        certificate = compute_catoni_bound(
            train_error=metrics_b["error"],
            divergence=divergence_nats,
            sample_size=metrics_b["count"],
            epsilon=args.delta,
        )
        bound_name = "catoni_fallback"

    if rank == 0:
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        checkpoint = {
            "model_state_dict": unwrapped.state_dict(),
            "dataset": args.dataset,
            "architecture": ARCHITECTURES[args.dataset],
            "seed": args.seed,
            "prior_fraction": args.prior_fraction,
            "intrinsic_dim": intrinsic_dim,
        }
        torch.save(checkpoint, output_dir / "prior_model.pt")
        np.save(output_dir / "split_indices.npy", permutation)

        results = {
            "method": "data-dependent compression PAC-Bayes",
            "dataset": args.dataset,
            "architecture": ARCHITECTURES[args.dataset],
            "seed": args.seed,
            "prior_fraction": args.prior_fraction,
            "n_A": len(indices_a),
            "n_B": len(indices_b),
            "intrinsic_dim": intrinsic_dim,
            "quantization_levels": 0,
            "message_bits": message_bits,
            "misc_extra_bits": args.misc_extra_bits,
            "divergence_nats": divergence_nats,
            "delta": args.delta,
            "confidence": 1.0 - args.delta,
            "bound": bound_name,
            "empirical_risk_B": metrics_b["error"],
            "certificate": float(certificate),
            "certificate_percent": 100.0 * float(certificate),
            "prior_train_A": metrics_a,
            "test_diagnostic": metrics_test,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": "cosine",
            "minimum_learning_rate": args.min_learning_rate,
            "epochs": args.epochs,
            "batch_size_per_process": args.batch_size,
            "world_size": world_size,
            "sync_batchnorm": bool(world_size > 1 and args.sync_bn),
            "data_root": str(Path(args.data_root).resolve()),
            "reportable": reportable,
            "elapsed_seconds": time.time() - started,
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps({
            "empirical_risk_B_percent": 100.0 * metrics_b["error"],
            "certificate_percent": 100.0 * float(certificate),
            "test_error_percent_diagnostic": 100.0 * metrics_test["error"],
            "output_dir": str(output_dir),
            "reportable": reportable,
        }, indent=2))

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
