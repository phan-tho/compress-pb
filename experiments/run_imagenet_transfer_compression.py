"""Compression PAC--Bayes transfer from ImageNet ResNet-18 to CIFAR.

This follows the original project's transfer protocol: an ImageNet-pretrained
backbone is held as the data-independent prior, its downstream classifier is
newly initialized, and all CIFAR parameters are learned in a seeded intrinsic
subspace before uniform quantization.  Unlike the split-prior experiment, no
CIFAR examples are used to construct the prior, so all 50,000 training images
are available for posterior training and certification.
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
import scipy.stats
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms

from pactl.benchmark_data import ArrayImageDataset, load_benchmark_arrays
from pactl.bounds.get_pac_bounds import compute_catoni_bound, pac_bayes_bound_opt
from pactl.bounds.quantize_fns import finetune_quantization, get_message_len
from pactl.nn.projectors import create_intrinsic_model


TRANSFER_DEFAULTS = {
    "cifar10": {"classes": 10, "dimension": 3000, "mode": "rdkron"},
    "cifar100": {"classes": 100, "dimension": 8000, "mode": "filmrdkron"},
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=tuple(TRANSFER_DEFAULTS), required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--weights-path", default=None,
                   help="Local torchvision ResNet-18 ImageNet weights.  If omitted, torchvision uses its cache or downloads them.")
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--intrinsic-dim", type=int, default=None)
    p.add_argument("--intrinsic-mode", choices=("rdkron", "filmrdkron"), default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Per-process batch size under torchrun.")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--quantization-epochs", type=int, default=30)
    p.add_argument("--quantization-learning-rate", type=float, default=3e-3)
    p.add_argument("--quantization-levels", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-train-examples", type=int, default=None,
                   help="For smoke tests only; marks the output non-reportable.")
    return p.parse_args()


def runtime(requested: str):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        if requested not in ("auto", "cuda") or not torch.cuda.is_available():
            raise RuntimeError("torchrun transfer training requires CUDA")
        torch.distributed.init_process_group("nccl", init_method="env://")
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), rank, world
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested), 0, 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cifar_transforms(train: bool):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


def loader(dataset, batch_size, workers, device, rank, world, shuffle, seed):
    sampler = None
    if world > 1:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                     shuffle=shuffle, seed=seed, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=workers, pin_memory=device.type == "cuda",
                      persistent_workers=workers > 0)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _hidden_modules(module):
    """Yield modules including projectors' deliberately unregistered nets."""
    yield module
    hidden = getattr(module, "_forward_net", None)
    if isinstance(hidden, list) and hidden:
        yield from _hidden_modules(hidden[0])


def freeze_batchnorm_statistics(model: nn.Module) -> None:
    # ImageNet BN running statistics are part of the prior, not posterior data.
    for root in _hidden_modules(unwrap(model)):
        for child in root.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.eval()


def all_reduce(value, world):
    if world > 1:
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return value


def train_epoch(model, data, optimizer, device, world):
    model.train()
    freeze_batchnorm_statistics(model)
    loss_total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for x, y in data:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        loss_total += loss.detach() * y.numel()
        count += y.numel()
    all_reduce(loss_total, world)
    all_reduce(count, world)
    return (loss_total / count).item()


@torch.inference_mode()
def evaluate(model, data, device, world=1):
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.float64)
    total = torch.zeros((), device=device, dtype=torch.float64)
    for x, y in data:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().double()
        total += y.numel()
    all_reduce(correct, world)
    all_reduce(total, world)
    return {"count": int(total.item()), "accuracy": (correct / total).item(),
            "error": 1.0 - (correct / total).item()}


def pretrained_resnet18(classes: int, weights_path: str | None) -> nn.Module:
    if weights_path:
        net = models.resnet18(weights=None)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        net.load_state_dict(state)
    else:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, classes)
    return net


def quantized_message(qw, levels: int):
    vector = qw.subspace_params.detach().cpu().numpy()
    codebook = qw.centroids.detach().cpu().numpy().astype(np.float16)
    symbols = np.argmin((vector[:, None] - codebook[None, :]) ** 2, axis=1)
    decoded = codebook[symbols].astype(np.float32)
    probabilities = np.bincount(symbols, minlength=levels) / len(symbols)
    entropy = scipy.stats.entropy(probabilities, base=2)
    coded_bits = float(math.ceil(len(symbols) * entropy) + 1)
    return decoded, codebook, symbols, get_message_len(coded_bits, codebook, len(symbols)), entropy


def main() -> None:
    args = parse_args()
    if not 0 < args.delta < 1:
        raise ValueError("--delta must be strictly between zero and one")
    device, rank, world = runtime(args.device)
    seed_everything(args.seed)
    defaults = TRANSFER_DEFAULTS[args.dataset]
    dim = args.intrinsic_dim or defaults["dimension"]
    mode = args.intrinsic_mode or defaults["mode"]
    out = Path(args.output_dir or f"runs/{args.dataset}-imagenet-resnet18-transfer-seed{args.seed}").resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        torch.distributed.barrier()
    started = time.time()

    tx, ty, vx, vy = load_benchmark_arrays(args.dataset, args.data_root)
    reportable = args.max_train_examples is None
    if args.max_train_examples is not None:
        tx, ty = tx[:args.max_train_examples], ty[:args.max_train_examples]
    train_set = ArrayImageDataset(tx, ty, cifar_transforms(train=True))
    certify_set = ArrayImageDataset(tx, ty, cifar_transforms(train=False))
    test_set = ArrayImageDataset(vx, vy, cifar_transforms(train=False))
    train_loader = loader(train_set, args.batch_size, args.num_workers, device, rank, world, True, args.seed)
    cert_loader = loader(certify_set, args.batch_size, args.num_workers, device, rank, world, False, args.seed)
    test_loader = loader(test_set, args.batch_size, args.num_workers, device, rank, world, False, args.seed)

    base = pretrained_resnet18(defaults["classes"], args.weights_path)
    if rank == 0:
        torch.save({"model_state_dict": base.state_dict(), "weights_path": args.weights_path,
                    "classifier": "new torch.nn.Linear"}, out / "imagenet_prior_and_new_head.pt")
    model = create_intrinsic_model(base, intrinsic_mode=mode, intrinsic_dim=dim, seed=args.seed).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if rank == 0:
        print(f"dataset={args.dataset} prior=ImageNet ResNet-18 d={dim} mode={mode} world_size={world}")
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, device, world)
        if rank == 0 and (epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.epochs):
            print(f"epoch={epoch + 1}/{args.epochs} train_nll={train_loss:.6f}")

    # Quantization is a deterministic rank-zero post-processing step; DDP has
    # already synchronized the learned intrinsic vector.
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        intrinsic = unwrap(model)
        qtrain = loader(train_set, args.batch_size, args.num_workers, device, 0, 1, True, args.seed)
        qw = finetune_quantization(intrinsic, args.quantization_levels, device, qtrain,
                                   args.quantization_epochs, nn.CrossEntropyLoss(), "adam",
                                   args.quantization_learning_rate, use_kmeans=False,
                                   freeze_batchnorm_statistics=True)
        freeze_batchnorm_statistics(qw)
        decoded, codebook, symbols, message_bits, entropy = quantized_message(qw, args.quantization_levels)
        qw.centroids.data.copy_(torch.as_tensor(codebook, device=device, dtype=qw.centroids.dtype))
        train_metrics = evaluate(qw, loader(certify_set, args.batch_size, args.num_workers, device, 0, 1, False, args.seed), device)
        test_metrics = evaluate(qw, loader(test_set, args.batch_size, args.num_workers, device, 0, 1, False, args.seed), device)
        divergence = message_bits * math.log(2.0)
        if train_metrics["accuracy"] >= 0.5:
            certificate, bound_name = pac_bayes_bound_opt(divergence, train_metrics["error"], train_metrics["count"], args.delta), "optimized_catoni"
        else:
            certificate, bound_name = compute_catoni_bound(divergence, train_metrics["error"], train_metrics["count"], args.delta), "catoni_fallback"
        torch.save({"subspace_params": torch.from_numpy(decoded), "codebook": torch.from_numpy(codebook),
                    "symbols": torch.from_numpy(symbols), "intrinsic_dim": dim, "intrinsic_mode": mode,
                    "projector_seed": args.seed}, out / "quantized_posterior.pt")
        results = {"method": "data-independent compression PAC-Bayes transfer", "dataset": args.dataset,
                   "architecture": "torchvision ResNet-18 (ImageNet pretrained; newly initialized CIFAR classifier)",
                   "prior_data": "ImageNet-1k", "posterior_data": "all CIFAR training examples", "seed": args.seed,
                   "intrinsic_dim": dim, "intrinsic_mode": mode, "quantization": {"levels": args.quantization_levels, "encoding": "arithmetic entropy estimate", "entropy_bits_per_symbol": entropy, "message_bits": message_bits},
                   "divergence_nats": divergence, "delta": args.delta, "bound": bound_name,
                   "empirical_risk_train": train_metrics["error"], "certificate": float(certificate), "certificate_percent": 100 * float(certificate),
                   "train_diagnostic": train_metrics, "test_diagnostic": test_metrics, "epochs": args.epochs,
                   "optimizer": "Adam", "learning_rate": args.learning_rate, "quantization_epochs": args.quantization_epochs,
                   "quantization_learning_rate": args.quantization_learning_rate, "batch_size_per_process": args.batch_size,
                   "world_size": world, "batchnorm_running_statistics": "frozen at ImageNet-prior values", "reportable": reportable,
                   "elapsed_seconds": time.time() - started, "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
        (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
        print(json.dumps({"certificate_percent": results["certificate_percent"], "test_error_percent_diagnostic": 100 * test_metrics["error"], "output_dir": str(out)}, indent=2))
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
