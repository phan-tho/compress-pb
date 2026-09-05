import gzip
import io
import pickle
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pactl.benchmark_data import load_benchmark_arrays
from pactl.paper_models import PaperMNISTCNN, PaperWideResNet, create_paper_model


class TestPaperModels(unittest.TestCase):
    def test_mnist_architecture_shape(self):
        model = PaperMNISTCNN()
        inputs = torch.randn(2, 1, 28, 28)
        self.assertEqual(model.forward_features(inputs).shape, (2, 32))
        self.assertEqual(model(inputs).shape, (2, 10))

    def test_wrn_architecture_shapes(self):
        for dataset, feature_dim, classes in (
            ("cifar10", 128, 10),
            ("cifar100", 256, 100),
        ):
            model = create_paper_model(dataset)
            self.assertIsInstance(model, PaperWideResNet)
            inputs = torch.randn(2, 3, 32, 32)
            self.assertEqual(model.forward_features(inputs).shape, (2, feature_dim))
            self.assertEqual(model(inputs).shape, (2, classes))
            self.assertEqual(len(model.group1), 4)
            self.assertEqual(len(model.group2), 4)
            self.assertEqual(len(model.group3), 4)


class TestMNISTReader(unittest.TestCase):
    @staticmethod
    def _write_images(path: Path, images: np.ndarray):
        payload = struct.pack(">IIII", 2051, *images.shape) + images.tobytes()
        with gzip.open(path, "wb") as handle:
            handle.write(payload)

    @staticmethod
    def _write_labels(path: Path, labels: np.ndarray):
        payload = struct.pack(">II", 2049, len(labels)) + labels.tobytes()
        with gzip.open(path, "wb") as handle:
            handle.write(payload)

    def test_kaggle_and_torchvision_idx_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
            labels = np.array([3, 7], dtype=np.uint8)
            self._write_images(root / "train-images.idx3-ubyte.gz", images)
            self._write_labels(root / "train-labels-idx1-ubyte.gz", labels)
            self._write_images(root / "t10k-images-idx3-ubyte.gz", images[:1])
            self._write_labels(root / "t10k-labels.idx1-ubyte.gz", labels[:1])
            train_x, train_y, test_x, test_y = load_benchmark_arrays("mnist", root)
            np.testing.assert_array_equal(train_x, images)
            np.testing.assert_array_equal(train_y, labels)
            np.testing.assert_array_equal(test_x, images[:1])
            np.testing.assert_array_equal(test_y, labels[:1])


class TestCIFARReaders(unittest.TestCase):
    @staticmethod
    def _flat_image(value: int):
        return np.full((1, 3 * 32 * 32), value, dtype=np.uint8)

    def test_cifar10_kaggle_tar(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "cifar-10-python.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for name, label, value in [
                    *[(f"data_batch_{i}", i, i) for i in range(1, 6)],
                    ("test_batch", 9, 9),
                ]:
                    payload = pickle.dumps({
                        b"data": self._flat_image(value), b"labels": [label]
                    })
                    info = tarfile.TarInfo(f"cifar-10-batches-py/{name}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            train_x, train_y, test_x, test_y = load_benchmark_arrays(
                "cifar10", directory
            )
            self.assertEqual(train_x.shape, (5, 32, 32, 3))
            np.testing.assert_array_equal(train_y, np.arange(1, 6))
            self.assertEqual(test_x.shape, (1, 32, 32, 3))
            np.testing.assert_array_equal(test_y, [9])

    def test_cifar100_kaggle_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, label, value in (("train", 12, 1), ("test", 34, 2)):
                with (root / name).open("wb") as handle:
                    pickle.dump({
                        b"data": self._flat_image(value),
                        b"fine_labels": [label],
                    }, handle)
            train_x, train_y, test_x, test_y = load_benchmark_arrays(
                "cifar100", directory
            )
            self.assertEqual(train_x.shape, (1, 32, 32, 3))
            np.testing.assert_array_equal(train_y, [12])
            self.assertEqual(test_x.shape, (1, 32, 32, 3))
            np.testing.assert_array_equal(test_y, [34])


if __name__ == "__main__":
    unittest.main()
