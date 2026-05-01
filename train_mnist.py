#!/usr/bin/env python3
"""
Train a TinyTorch CNN on MNIST using the Python bindings (base, nn, optim, mnist_io).

Requires CUDA, built extensions on PYTHONPATH from this repo root, and IDX files under data/:

  data/train-images.idx3-ubyte
  data/train-labels.idx1-ubyte

Each forward allocates a new subgraph; long runs reuse a lot of host/device memory inside the
compiled library. Prefer modest --epochs / --limit-batches for smoke tests unless you rebuild
with explicit tensor lifecycle management.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def import_extensions():
    """Extensions are emitted next to CMake bind output (typically this directory)."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def argmax_logits_row(logits_data_on_host: np.ndarray, n_classes: int, batch_idx: int) -> int:
    row = logits_data_on_host[batch_idx * n_classes : (batch_idx + 1) * n_classes]
    return int(np.argmax(row))


def run_epoch(base, optim, data_io, dataset, ix_list, n_samples, batch_size, conv, relu, pool, lin, ce, optimizer, shuffle: bool):
    ix = ix_list[:] if shuffle else ix_list
    if shuffle:
        np.random.shuffle(ix)

    num_batches = n_samples // batch_size
    total_loss = 0.0
    correct = 0
    total_seen = 0

    imgs = base.tensor_create([batch_size, 1, 28, 28], 0, 0)
    lbl_cpu = base.tensor_create([batch_size], 0, 0)

    try:
        for bi in range(num_batches):
            s = bi * batch_size
            e = s + batch_size
            base.tensor_to_cpu(imgs)
            base.tensor_to_cpu(lbl_cpu)
            data_io.load_batch_to_tensor(dataset, s, e, ix, imgs, lbl_cpu)
            base.tensor_to_gpu(imgs)
            base.tensor_to_gpu(lbl_cpu)

            h1 = conv.forward(imgs)
            h2 = relu.forward(h1)
            
            # h3 = conv.forward(h2)
            # h4 = relu.forward(h3)
            
            h3 = pool.forward(h2)

            logits = lin.forward(h3)
            loss = ce.forward(logits, lbl_cpu)

            base.tensor_to_cpu(loss)
            total_loss += float(loss.data[0])

            optim.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            base.tensor_to_cpu(logits)
            base.tensor_to_cpu(lbl_cpu)
            ld = np.array(logits.data, copy=True)
            yd = np.array(lbl_cpu.data, copy=True)

            bs = batch_size
            nc = logits.shape[-1]
            for j in range(bs):
                pred = argmax_logits_row(ld, nc, j)
                if pred == int(round(yd[j])):
                    correct += 1
            total_seen += bs

            base.tensor_free(h1)
            base.tensor_free(h2)
            base.tensor_free(h3)
            # base.tensor_free(h4)
            # base.tensor_free(h5)
            base.tensor_free(logits)
            base.tensor_free(loss)

    finally:
        base.tensor_free(imgs)
        base.tensor_free(lbl_cpu)

    avg_loss = total_loss / max(1, num_batches)
    acc = correct / max(1, total_seen)
    return avg_loss, acc


def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="MNIST trainer (TinyTorch pybind)")
    p.add_argument("--data-dir", type=pathlib.Path, default=None, help="Folder with IDX files (default: <repo>/data)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="If set on train set, restrict to first N batches per epoch (smaller allocations).",
    )
    args = p.parse_args(argv)

    root = repo_root()
    data_dir = args.data_dir.resolve() if args.data_dir else (root / "data")
    train_img = data_dir / "train-images.idx3-ubyte"
    train_lbl = data_dir / "train-labels.idx1-ubyte"

    if not train_img.is_file() or not train_lbl.is_file():
        print(f"Missing MNIST IDX under {data_dir}", file=sys.stderr)
        sys.exit(1)

    import_extensions()

    np.random.seed(args.seed)

    import base
    import mnist_io as data
    import nn
    import optim as optim_mod

    n_train = 30000
    train_ds = data.load_dataset_in_ram(str(train_img), str(train_lbl), n_train)
    ix_train = np.array(data.create_indices(n_train), dtype=np.int32).tolist()

    n_eff = n_train
    if args.limit_batches is not None:
        n_eff = min(n_train, args.limit_batches * args.batch_size)
    bs = args.batch_size
    if n_eff % bs != 0:
        print(f"Warning: truncating dataset to {(n_eff // bs) * bs} samples (full batches only)", file=sys.stderr)
        n_eff = (n_eff // bs) * bs

    # Conv(padding nonzero => same H,W as input in this codebase), ReLU, global pool -> [B,C], Linear -> logits
    conv = nn.Conv2D(1, 32, 3, 1, 1)
    relu = nn.ReLU(1)
    pool = nn.GlobalPooling(1)
    lin = nn.Linear(32, 10, 1, 1)
    ce = nn.CrossEntropy(1)

    params = [conv.weights, lin.weights, lin.bias]
    optimizer = optim_mod.SGD(params, args.lr)

    if args.limit_batches is not None:
        print(f"limit-batches: using {n_eff} samples per epoch (not full MNIST)")
    print(f"Training samples per epoch={n_eff}, batch_size={bs}, lr={args.lr}")

    for ep in range(args.epochs):
        loss_avg, acc = run_epoch(
            base,
            optim_mod,
            data,
            train_ds,
            ix_train[:n_eff],
            n_eff,
            bs,
            conv,
            relu,
            pool,
            lin,
            ce,
            optimizer,
            shuffle=True,
        )
        print(f"epoch {ep + 1}/{args.epochs}  avg_loss={loss_avg:.4f}  train_acc={acc:.4f}")

    data.free_mnist_data(train_ds)
    print("Done.")


if __name__ == "__main__":
    main()
