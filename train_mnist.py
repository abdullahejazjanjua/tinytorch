from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, List, Tuple

import numpy as np

NUM_CLASSES = 10


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def import_extensions():
    """Extensions are emitted next to CMake bind output (typically this directory)."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def argmax_logits_row(logits_host: np.ndarray, n_classes: int, batch_idx: int) -> int:
    row = logits_host[batch_idx * n_classes : (batch_idx + 1) * n_classes]
    return int(np.argmax(row))


def build_mnist_backbone(
    nn_mod: Any,
    *,
    shallow: bool,
    c1: int,
    c2: int,
    head_hidden: int,
) -> Tuple[List[Any], List[Any], str]:
    """Construct ``backbone`` and ``optimizer_param_tensors`` with consistent tensor shapes.

    - *shallow*: single ``conv→relu→pool`` stem (legacy baseline).
    - Otherwise two conv blocks ``1→c1→c2`` then global pool ``[N,c2,H,W]→[N,c2]``.
    - ``head_hidden > 0`` adds ``Linear→ReLU→Linear`` logits head; ``0`` is a single ``Linear``

    Returns ``(backbone, params, architecture_description)``
    """

    backbone: List[Any] = []
    params: List[Any] = []

    stem_desc: str
    conv_in = c1 if shallow else c2

    if shallow:
        conv1 = nn_mod.Conv2D(1, c1, 5, 1, 1)
        relu_s = nn_mod.ReLU(1)
        pool = nn_mod.GlobalPooling(1)
        backbone.extend([conv1, relu_s, pool])
        params.append(conv1.weights)
        stem_desc = f"stem:1→{c1}ch Conv, pool"
    else:
        conv1 = nn_mod.Conv2D(1, c1, 5, 1, 1)
        r1 = nn_mod.ReLU(1)
        conv2 = nn_mod.Conv2D(c1, c2, 5, 1, 1)
        r2 = nn_mod.ReLU(1)
        pool = nn_mod.GlobalPooling(1)
        backbone.extend([conv1, r1, conv2, r2, pool])
        params.extend([conv1.weights, conv2.weights])
        stem_desc = f"stem:1→{c1}→{c2}ch Conv, pool"

    if head_hidden > 0:
        lin1 = nn_mod.Linear(conv_in, head_hidden, 1, 1)
        hr = nn_mod.ReLU(1)
        lin2 = nn_mod.Linear(head_hidden, NUM_CLASSES, 1, 1)
        backbone.extend([lin1, hr, lin2])
        params.extend([lin1.weights, lin1.bias, lin2.weights, lin2.bias])
        head_desc = f"head:{conv_in}→{head_hidden}→{NUM_CLASSES}"
    else:
        lin = nn_mod.Linear(conv_in, NUM_CLASSES, 1, 1)
        backbone.append(lin)
        params.extend([lin.weights, lin.bias])
        head_desc = f"head:{conv_in}→{NUM_CLASSES}"

    desc = stem_desc + ";" + head_desc
    return backbone, params, desc


def run_epoch(
    base: Any,
    optim: Any,
    data_io: Any,
    dataset: Any,
    ix_list: List[int],
    n_samples: int,
    batch_size: int,
    backbone: List[Any],
    ce: Any,
    optimizer: Any,
    shuffle: bool,
    *,
    epoch_no: int,
    epoch_total: int,
    running_loss_every: int,
):
    """Run one epoch: *backbone* ends with logits ``[batch, NUM_CLASSES]`` (GEMM-aligned)."""

    ix = ix_list[:] if shuffle else ix_list
    if shuffle:
        np.random.shuffle(ix)

    num_batches = n_samples // batch_size
    total_loss = 0.0
    correct = 0
    total_seen = 0

    imgs = base.tensor_create([batch_size, 1, 28, 28], 0, 0)
    lbl_cpu = base.tensor_create([batch_size], 0, 0)

    print(f"epoch {epoch_no}/{epoch_total}: ")

    try:
        for bi in range(num_batches):
            s = bi * batch_size
            e = s + batch_size
            base.tensor_to_cpu(imgs)
            base.tensor_to_cpu(lbl_cpu)
            data_io.load_batch_to_tensor(dataset, s, e, ix, imgs, lbl_cpu)
            base.tensor_to_gpu(imgs)
            base.tensor_to_gpu(lbl_cpu)

            activations_to_free = []
            x = imgs
            for layer in backbone:
                x = layer.forward(x)
                activations_to_free.append(x)

            logits = activations_to_free[-1]

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
            nc = logits.shape[-1]

            bs = batch_size
            for j in range(bs):
                pred = argmax_logits_row(ld, nc, j)
                if pred == int(round(yd[j])):
                    correct += 1
            total_seen += bs

            base.tensor_free(loss)
            for t in reversed(activations_to_free):
                base.tensor_free(t)

            batches_done = bi + 1
            running_avg = total_loss / batches_done
            if running_loss_every > 0:
                step_done = batches_done % running_loss_every == 0
                epoch_end = batches_done == num_batches
                if step_done or epoch_end:
                    print(
                        f"  batch {batches_done}/{num_batches}  "
                        f"  running_loss={running_avg:.6f}",
                        flush=True,
                    )
    finally:
        base.tensor_free(imgs)
        base.tensor_free(lbl_cpu)

    avg_loss = total_loss / max(1, num_batches)
    acc = correct / max(1, total_seen)
    print(
        f"epoch {epoch_no}/{epoch_total}  summary  avg_loss={avg_loss:.6f}  train_acc={acc:.4f}",
        flush=True,
    )
    return avg_loss, acc


def evaluate_classifier(
    base: Any,
    data_io: Any,
    dataset: Any,
    ix_list: List[int],
    n_samples: int,
    batch_size: int,
    backbone: List[Any],
    ce: Any,
    *,
    label: str,
) -> Tuple[float, float]:
    """Forward-only: mean minibatch CE and accuracy on fixed full batches."""

    n_eval = (n_samples // batch_size) * batch_size
    num_batches = n_eval // batch_size
    if num_batches <= 0:
        return 0.0, 0.0

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
            data_io.load_batch_to_tensor(dataset, s, e, ix_list, imgs, lbl_cpu)
            base.tensor_to_gpu(imgs)
            base.tensor_to_gpu(lbl_cpu)

            activations_to_free = []
            x = imgs
            for layer in backbone:
                x = layer.forward(x)
                activations_to_free.append(x)

            logits = activations_to_free[-1]

            loss = ce.forward(logits, lbl_cpu)
            base.tensor_to_cpu(loss)
            total_loss += float(loss.data[0])

            base.tensor_to_cpu(logits)
            base.tensor_to_cpu(lbl_cpu)
            ld = np.array(logits.data, copy=True)
            yd = np.array(lbl_cpu.data, copy=True)
            nc = logits.shape[-1]

            bs = batch_size
            for j in range(bs):
                pred = argmax_logits_row(ld, nc, j)
                if pred == int(round(yd[j])):
                    correct += 1
            total_seen += bs

            base.tensor_free(loss)
            for t in reversed(activations_to_free):
                base.tensor_free(t)
    finally:
        base.tensor_free(imgs)
        base.tensor_free(lbl_cpu)

    avg_loss = total_loss / num_batches
    acc = correct / max(1, total_seen)
    print(
        f"{label}  examples={n_eval}  batches={num_batches}  "
        f"avg_loss={avg_loss:.6f}  accuracy={acc:.4f}",
        flush=True,
    )
    return avg_loss, acc


def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="MNIST trainer (TinyTorch pybind)")
    p.add_argument("--data-dir", type=pathlib.Path, default=None, help="Folder with IDX files (default: <repo>/data)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Smaller batch (e.g. 32) lowers peak VRAM if needed.",
    )
    p.add_argument(
        "--shallow",
        action="store_true",
        help="Legacy singleConv+pool stem (fewer params, slower to fit than the default stem).",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=48,
        help="Stem: output channels after first conv.",
    )
    p.add_argument(
        "--channels-last",
        type=int,
        default=96,
        help="Stem (non-shallow): second conv output channels = width before pool / classifier head.",
    )
    p.add_argument(
        "--head-hidden",
        type=int,
        default=128,
        help="MLP hidden size after global pool before logits (0 disables MLP → single Linear).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="SGD learning rate.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="If set, restrict to first N batches per epoch.",
    )
    p.add_argument(
        "--running-loss-every",
        type=int,
        default=100,
        help="Print running mean minibatch CE loss every N batches within an epoch; 0 = only epoch summary.",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run forward-only accuracy on MNIST test (t10k) after training.",
    )
    p.add_argument("--two-conv", action="store_true", help=argparse.SUPPRESS)

    args = p.parse_args(argv)

    if args.channels < 1:
        p.error("--channels must be positive")
    if not args.shallow and args.channels_last < 1:
        p.error("--channels-last must be positive when not using --shallow")
    if args.head_hidden < 0:
        p.error("--head-hidden must be non-negative")
    if args.running_loss_every < 0:
        p.error("--running-loss-every must be non-negative")

    root = repo_root()
    data_dir = args.data_dir.resolve() if args.data_dir else (root / "data")
    train_img = data_dir / "train-images.idx3-ubyte"
    train_lbl = data_dir / "train-labels.idx1-ubyte"
    test_img = data_dir / "t10k-images.idx3-ubyte"
    test_lbl = data_dir / "t10k-labels.idx1-ubyte"

    if not train_img.is_file() or not train_lbl.is_file():
        print(f"Missing MNIST IDX under {data_dir}", file=sys.stderr)
        sys.exit(1)

    import_extensions()

    np.random.seed(args.seed)

    import base
    import mnist_io as data
    import nn
    import optim as optim_mod

    backbone, params, arch_desc = build_mnist_backbone(
        nn,
        shallow=args.shallow,
        c1=args.channels,
        c2=args.channels_last,
        head_hidden=args.head_hidden,
    )

    ce = nn.CrossEntropy(1)
    optimizer = optim_mod.SGD(params, args.lr)

    n_train = 60_000
    train_ds = data.load_dataset_in_ram(str(train_img), str(train_lbl), n_train)
    ix_train = np.array(data.create_indices(n_train), dtype=np.int32).tolist()

    n_eff = n_train
    if args.limit_batches is not None:
        n_eff = min(n_train, args.limit_batches * args.batch_size)
    bs = args.batch_size
    if n_eff % bs != 0:
        print(f"Warning: truncating dataset to {(n_eff // bs) * bs} samples (full batches only)", file=sys.stderr)
        n_eff = (n_eff // bs) * bs

    if args.limit_batches is not None:
        print(f"limit-batches: using {n_eff} samples per epoch (not full MNIST)")
    print(f"samples/epoch={n_eff} bs={bs} lr={args.lr}  {arch_desc}")

    for ep in range(args.epochs):
        run_epoch(
            base,
            optim_mod,
            data,
            train_ds,
            ix_train[:n_eff],
            n_eff,
            bs,
            backbone,
            ce,
            optimizer,
            shuffle=True,
            epoch_no=ep + 1,
            epoch_total=args.epochs,
            running_loss_every=args.running_loss_every,
        )

    data.free_mnist_data(train_ds)

    if args.skip_eval:
        print("Done (eval skipped).", flush=True)
        return

    if not test_img.is_file() or not test_lbl.is_file():
        print(
            f"Skipping evaluation: missing test IDX under {data_dir}. "
            f"Put t10k-images.idx3-ubyte and t10k-labels.idx1-ubyte there.",
            file=sys.stderr,
        )
        print("Done.", flush=True)
        return

    n_test_set = 10_000
    test_ds = data.load_dataset_in_ram(str(test_img), str(test_lbl), n_test_set)
    ix_test = data.create_indices(n_test_set)

    evaluate_classifier(
        base,
        data,
        test_ds,
        ix_test,
        n_test_set,
        bs,
        backbone,
        ce,
        label="Testing:",
    )

    data.free_mnist_data(test_ds)
    print("Done.")


if __name__ == "__main__":
    main()