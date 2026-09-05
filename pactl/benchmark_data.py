"""Read the local and Kaggle copies of MNIST, CIFAR-10, and CIFAR-100."""

from __future__ import annotations

import gzip
import pickle
import struct
import tarfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


NORMALIZATION = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


class ArrayImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        if len(images) != len(labels):
            raise ValueError("Image and label counts do not match")
        self.images = images
        self.labels = labels.astype(np.int64, copy=False)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def make_transform(dataset: str, augment: bool):
    operations = []
    if augment and dataset.startswith("cifar"):
        operations.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    mean, std = NORMALIZATION[dataset]
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(operations)


def _find_file(root: Path, names: Iterable[str]) -> Path:
    names = tuple(names)
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for path in root.rglob("*"):
        if path.is_file() and path.name in names:
            return path
    raise FileNotFoundError(f"Could not find any of {names} below {root}")


def _read_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def _read_idx_images(path: Path) -> np.ndarray:
    raw = _read_bytes(path)
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Invalid IDX image file {path}: magic={magic}")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    raw = _read_bytes(path)
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Invalid IDX label file {path}: magic={magic}")
    return np.frombuffer(raw, dtype=np.uint8, offset=8, count=count).copy()


def _mnist_file(root: Path, stem: str) -> Path:
    dotted = stem.replace("-idx", ".idx")
    return _find_file(root, (stem, f"{stem}.gz", dotted, f"{dotted}.gz"))


def _load_mnist(root: Path):
    train_images = _read_idx_images(_mnist_file(root, "train-images-idx3-ubyte"))
    train_labels = _read_idx_labels(_mnist_file(root, "train-labels-idx1-ubyte"))
    test_images = _read_idx_images(_mnist_file(root, "t10k-images-idx3-ubyte"))
    test_labels = _read_idx_labels(_mnist_file(root, "t10k-labels-idx1-ubyte"))
    return train_images, train_labels, test_images, test_labels


def _pickle_dict(handle):
    return pickle.load(handle, encoding="bytes")


def _pickle_path(path: Path):
    with path.open("rb") as handle:
        return _pickle_dict(handle)


def _load_cifar10(root: Path):
    try:
        batch_paths = [_find_file(root, (f"data_batch_{i}",)) for i in range(1, 6)]
        test_path = _find_file(root, ("test_batch",))
        batches = [_pickle_path(path) for path in batch_paths]
        test = _pickle_path(test_path)
    except FileNotFoundError:
        archive_path = _find_file(
            root, ("cifar-10-python.tar.gz", "cifar-10-python.tar")
        )
        with tarfile.open(archive_path, "r:*") as archive:
            def member(name: str):
                extracted = archive.extractfile(f"cifar-10-batches-py/{name}")
                if extracted is None:
                    raise FileNotFoundError(name)
                return _pickle_dict(extracted)

            batches = [member(f"data_batch_{i}") for i in range(1, 6)]
            test = member("test_batch")

    train_x = np.concatenate([batch[b"data"] for batch in batches])
    train_y = np.concatenate([np.asarray(batch[b"labels"]) for batch in batches])
    test_x = test[b"data"]
    test_y = np.asarray(test[b"labels"])
    return _reshape_cifar(train_x), train_y, _reshape_cifar(test_x), test_y


def _load_cifar100(root: Path):
    train_path = _find_file(root, ("train",))
    test_path = _find_file(root, ("test",))
    with train_path.open("rb") as handle:
        train = _pickle_dict(handle)
    with test_path.open("rb") as handle:
        test = _pickle_dict(handle)
    return (
        _reshape_cifar(train[b"data"]), np.asarray(train[b"fine_labels"]),
        _reshape_cifar(test[b"data"]), np.asarray(test[b"fine_labels"]),
    )


def _reshape_cifar(array: np.ndarray) -> np.ndarray:
    return array.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def load_benchmark_arrays(dataset: str, root: str | Path):
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if dataset == "mnist":
        return _load_mnist(root)
    if dataset == "cifar10":
        return _load_cifar10(root)
    if dataset == "cifar100":
        return _load_cifar100(root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def make_benchmark_datasets(dataset: str, root: str | Path):
    train_x, train_y, test_x, test_y = load_benchmark_arrays(dataset, root)
    return (
        ArrayImageDataset(train_x, train_y, make_transform(dataset, augment=True)),
        ArrayImageDataset(train_x, train_y, make_transform(dataset, augment=False)),
        ArrayImageDataset(test_x, test_y, make_transform(dataset, augment=False)),
    )
