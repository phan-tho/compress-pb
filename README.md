# Tight PAC-Bayes Compression Bounds

[![](https://img.shields.io/badge/arXiv-2211.xxxxx-red)]() [![](https://img.shields.io/badge/NeurIPS-2022-green)]()

This repository hosts the code for [PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization]() by [Sanae Lotfi*](https://sanaelotfi.github.io), [Marc Finzi*](https://mfinzi.github.io), [Sanyam Kapoor*](https://sanyamkapoor.com), [Andres Potapczynski*](https://www.andpotap.com), [Micah Goldblum](https://goldblum.github.io), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

## Setup

```shell
conda env create -f environment.yml -n pactl
```

Setup the `pactl` package.

```shell
pip install -e .
```

## Usage

We use [Fire](https://google.github.io/python-fire/guide/) for CLI parsing.

### Matched split-prior benchmark

`experiments/run_split_compression.py` runs the data-dependent compression
baseline with the architectures used by the output-space PAC--Bayes comparison:

* MNIST: two 5x5 convolution/max-pool blocks followed directly by a linear
  classifier.
* CIFAR-10 and CIFAR-100: standard pre-activation WRN-28-4 with global average
  pooling and its standard linear classifier.

The baseline does not use the output-space method's projected feature map,
LayerNorm, tanh transformation, or rank parameter. The compression method is
applied to the standard classifier architecture on its own terms.

The split is a seeded permutation of the official training set. Subset A is
used to train the data-dependent prior and subset B is never read until the
prior has been frozen. The default matched run uses a 50/50 split, 200 epochs
of Adam, cosine learning-rate decay from 0.001 to 0.00001, intrinsic dimension
zero, no quantization, and a 95% optimized Catoni certificate. The source
paper reports that intrinsic dimension zero was the
selected configuration for every data-dependent dataset (Appendix E.2,
Table 7). Since this runner predeclares that single configuration, it incurs no
hyperparameter-search bits. Use `--misc-extra-bits` if comparing several
configurations and selecting one after observing subset B.

The runner reads the Kaggle dataset layouts directly, including the CIFAR-10
tar archive and the raw MNIST IDX files. A two-GPU run uses DDP and synchronized
BatchNorm. For an effective global batch size of 128, pass a per-process batch
size of 64:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m experiments.run_split_compression \
  --dataset cifar10 \
  --data-root /kaggle/input/datasets/pankrzysiu/cifar10-python \
  --output-dir /kaggle/working/compress-pb-results/cifar10 \
  --batch-size 64
```

The output directory contains `results.json`, `run_config.json`, the exact
split permutation, and the frozen prior checkpoint. Test error in
`results.json` is diagnostic only. Runs using `--max-train-examples` are marked
`"reportable": false` and are intended only as smoke tests.

### ImageNet ResNet-18 transfer benchmark

`experiments/run_imagenet_transfer_compression.py` is the separate
data-independent transfer baseline for CIFAR-10 and CIFAR-100.  It starts with
the official torchvision ImageNet-1k ResNet-18 weights, replaces only the
final `fc` layer for the downstream number of classes, and trains every model
parameter through the source method's seeded structured intrinsic subspace.
It resizes CIFAR images to 224 pixels and uses ImageNet normalization. ImageNet
BatchNorm running statistics stay frozen as part of the prior.

The default source-style configurations are: CIFAR-10 uses `d=3000` with the
`rdkron` projector; CIFAR-100 uses `d=8000` with `filmrdkron`. Both use 500
epochs of Adam at 0.001 followed by 30 epochs of seven-level uniform
quantization fine-tuning at 0.003. The CIFAR training set is not split: the
prior is independent because it was trained on ImageNet.

For Kaggle with two T4 GPUs, attach a dataset containing the official
`resnet18-f37072fd.pth` checkpoint and set `--weights-path` to its mounted
path. This avoids relying on an Internet download during the notebook run:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m experiments.run_imagenet_transfer_compression \
  --dataset cifar10 \
  --data-root /kaggle/input/datasets/pankrzysiu/cifar10-python \
  --weights-path /kaggle/input/<resnet18-weights>/resnet18-f37072fd.pth \
  --output-dir /kaggle/working/compress-pb-results/cifar10-imagenet-r18 \
  --batch-size 64
```

Use the same command with `--dataset cifar100`, the CIFAR-100 data root, and a
different output directory. The output contains the ImageNet prior plus new
head, the exactly decoded quantized posterior, and `results.json`. The test
metric is a diagnostic; the certificate uses all CIFAR training examples.

### Training Intrinsic Dimensionality Models


```shell
python experiments/train.py --dataset=cifar10 \
                            --model-name=resnet18k \
                            --base-width=64 \
                            --optimizer=adam \
                            --epochs=500 \
                            --lr=1e-3 \
                            --intrinsic_dim=1000 \
                            --intrinsic_mode=rdkronqr \
                            --seed=137
```

All arguments in the `main` method of [experiments/train.py](./experiments/train.py)
are valid CLI arguments. The most imporant ones are noted here:

* `--seed`: Setting the seed is important so that any subsequent runs using the checkpoint can reconstruct the same random parameter projection matrices used during training.
* `--data_dir`: Parent path to directory containing root directory of the dataset.
* `--dataset`: Dataset name. See [data.py](./pactl/data.py) for list of dataset strings.
* `--intrinsic_dim`: Dimension of the training subspace of parameters.
* `--intrinsic_mode`: Method used to generate (sparse) random projection matrices. See `create_intrinsic_model` method in [projectors.py](./pactl/nn/projectors.py) for a list of valid modes.

#### Distributed Training

Distributed training is helpful for large datasets like Imagenet to spread computation over multiple GPUs. 
We rely on [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

To use multiple GPUs on a single node, we need:
* GPU visibility flags appropriately via `CUDA_VISIBLE_DEVICES`.
* Specify the number `x` of GPUs made visible via `--nproc_per_node=<x>`
* Specify a random port `yyyy` on the host for inter-process communication via `--rdzv_endpoint=localhost:yyyy`.

For the same run as above, we simply replace `python` with `torchrun` as:
```shell
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:9999 experiments/train.py ...
```
All remaining CLI arguments remain unchanged.

### Transfer Learning using Existing Checkpoints

The key argument needed for transfer is the path to the configuration file named `net.cfg.yml` of the pretrained network. 

```shell
python experiments/train.py --dataset=fmnist \
                            --optimizer=adam \
                            --epochs=500 \
                            --lr=1e-3 \
                            --intrinsic_dim=1000 \
                            --intrinsic_mode=rdkronqr \
                            --prenet_cfg_path=<path/to/net.cfg.yml> \
                            --seed=137 \
                            --transfer
```

In addition to earlier arguments, there is only one new key argument:
* `--prenet_cfg_path`: Path to `net.cfg.yml` configuration file of the pretrained network. This path is logged during the train command specified previously.

### Training for Data-Dependent Bounds

Data-dependent bounds first require pre-training on a fixed subset of training data and then training
an intrinsic dimensionality model on the remainder of the subset.

For such training, we use the following command:
```shell
python experiments/train_dd_priors.py --dataset=cifar10 \
                                      ...
                                      --indices_path=<path/to/index/list> \
                                      --train-subset=0.1 \
                                      --seed=137
```

The key new arguments here in addition to the ones seen previously are:
* `--indices-path`: A fixed permutation of indices as a numpy list equal to the length of the dataset. If not specified, a random permutation is generated every time and the results may not be reproducible. See [dataset_permutations.ipynb](./notebooks/dataset_permutations.ipynb) to see an example of how to generate such a file.
* `--train-subset`: A fractional subset of the training data to use. If a negative fraction, then the complement is used.

### Computing our Adaptive Compression Bounds

Once we have the checkpoints of intrinsic-dimensionality models, the bound can be computed using:

```shell
python experiments/compute_bound.py --dataset=mnist \
                                    --misc-extra-bits=7 \
                                    --quant-epochs=30 \
                                    --levels=50 \
                                    --lr=0.0001 \
                                    --prenet_cfg_path=<path/to/net.cfg.yml> \
                                    --use_kmeans=True
```

The key arguments here are:
* `--misc-extra-bit`: Penalty for hyper-parameter optimization during bound computation, equals the bits required to encode all hyper-parameter configurations.
* `--levels`: Number of quantization levels.
* `--quant-epochs`: Number of epochs used for fine-tuning of quantization levels.
* `--lr`: Learning rate used for fine-tuning of quantization levels.
* `--user_kmeans`: When true, uses kMeans clustering for initialization of quantization levels. Otherwise, random initialization is used.

## LICENSE

Apache 2.0
