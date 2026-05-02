from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import MNIST
except ImportError as e:
    print("Requires PyTorch + torchvision with CUDA: pip install torch torchvision", file=sys.stderr)
    raise e

NUM_CLASSES = 10


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def trim_to_batches(n: int, batch_size: int) -> int:
    return (n // batch_size) * batch_size


class MnistNet(nn.Module):
    """Architectural twin of TinyTorch ``build_mnist_backbone``: Conv 5×5, padding 2, AdaptiveAvgPool2d(1)."""

    def __init__(self, *, shallow: bool, c1: int, c2: int, head_hidden: int):
        super().__init__()
        self.head_hidden = head_hidden
        if shallow:
            self.conv1 = nn.Conv2d(1, c1, kernel_size=5, padding=2, bias=False)
            self.conv2 = None
            stem = c1
        else:
            self.conv1 = nn.Conv2d(1, c1, kernel_size=5, padding=2, bias=False)
            self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, padding=2, bias=False)
            stem = c2
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if head_hidden > 0:
            self.fc1 = nn.Linear(stem, head_hidden)
            self.fc2 = nn.Linear(head_hidden, NUM_CLASSES)
            self.fc_out = None
        else:
            self.fc1 = self.fc2 = None
            self.fc_out = nn.Linear(stem, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu_(self.conv1(x))
        if self.conv2 is not None:
            x = torch.relu_(self.conv2(x))
        x = self.pool(x).flatten(1)
        if self.head_hidden > 0:
            x = torch.relu_(self.fc1(x))
            return self.fc2(x)
        return self.fc_out(x)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    epoch_no: int,
    epoch_total: int,
    device: torch.device,
    *,
    running_loss_every: int,
) -> Tuple[float, float, float]:
    """Return (avg_loss, train_acc, elapsed_seconds)."""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    batches = 0

    print()
    print("Epoch", epoch_no, "of", epoch_total)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        li = loss.detach().item()
        batches += 1
        bs = batch_x.size(0)
        total_loss += li
        pred = logits.detach().argmax(dim=1)
        total_correct += (pred == batch_y).sum().item()
        total_n += bs

        running = total_loss / batches
        if running_loss_every > 0:
            step_done = batches % running_loss_every == 0
            last = batches == len(loader)
            if step_done or last:
                print(
                    "  batch",
                    batches,
                    "of",
                    len(loader),
                    "| running loss:",
                    round(running, 6),
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    avg_loss = total_loss / max(1, batches)
    acc = total_correct / max(1, total_n)
    print()
    print(
        "Epoch",
        epoch_no,
        "summary | avg loss:",
        round(avg_loss, 6),
        "| train accuracy:",
        round(acc, 4),
        "| time (s):",
        round(elapsed, 3),
    )
    return avg_loss, acc, elapsed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    label: str,
) -> Tuple[float, float]:
    """Loader covers only full-batch ``Subset`` sizing."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    batches = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        batches += 1
        bs = batch_x.size(0)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        total_correct += (pred == batch_y).sum().item()
        total_n += bs

    if device.type == "cuda":
        torch.cuda.synchronize()

    avg_loss = total_loss / max(1, batches)
    acc = total_correct / max(1, total_n)
    examples = batches * loader.batch_size
    print()
    print(label)
    print("  samples:", examples, "| batches:", batches)
    print("  avg loss:", round(avg_loss, 6), "| accuracy:", round(acc, 4))
    return avg_loss, acc


def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="MNIST CUDA trainer (PyTorch MNIST dataset)")
    p.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=None,
        help="Root folder for torchvision MNIST cache (creates MNIST/ under it; default: <repo>/data)",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Do not fetch MNIST; files must exist under MNIST/processed/",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--shallow", action="store_true")
    p.add_argument("--channels", type=int, default=48)
    p.add_argument("--channels-last", type=int, default=96)
    p.add_argument("--head-hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-batches", type=int, default=None)
    p.add_argument("--running-loss-every", type=int, default=100)
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA required for this script.", file=sys.stderr)
        sys.exit(1)

    if args.channels < 1:
        p.error("--channels must be positive")
    if not args.shallow and args.channels_last < 1:
        p.error("--channels-last must be positive when not using --shallow")
    if args.head_hidden < 0:
        p.error("--head-hidden must be non-negative")
    if args.running_loss_every < 0:
        p.error("--running-loss-every must be non-negative")

    root = repo_root()
    data_dir = str((args.data_dir.resolve() if args.data_dir else (root / "data")))

    torch.manual_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    device = torch.device("cuda")

    tfm = T.ToTensor()

    train_full = MNIST(root=data_dir, train=True, download=not args.no_download, transform=tfm)

    n_full = len(train_full)
    n_eff = trim_to_batches(n_full, args.batch_size)
    if args.limit_batches is not None:
        capped = args.limit_batches * args.batch_size
        n_eff = min(n_eff, trim_to_batches(capped, args.batch_size))

    indices = list(range(n_eff))
    train_ds = Subset(train_full, indices)

    model = MnistNet(
        shallow=args.shallow,
        c1=args.channels,
        c2=args.channels_last,
        head_hidden=args.head_hidden,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=1,
        generator=g,
    )

    conv_in = args.channels if args.shallow else args.channels_last
    if args.head_hidden > 0:
        head_part = f"head:{conv_in}→{args.head_hidden}→{NUM_CLASSES}"
    else:
        head_part = f"head:{conv_in}→{NUM_CLASSES}"
    stem_part = (
        f"stem:1→{args.channels}ch Conv, pool"
        if args.shallow
        else f"stem:1→{args.channels}→{args.channels_last}ch Conv, pool"
    )
    arch_tag = stem_part + ";" + head_part

    print()
    print("Training config")
    print("  samples per epoch:", n_eff)
    print("  batch size:", args.batch_size)
    print("  learning rate:", args.lr)
    print("  data root:", data_dir)
    print("  device:", "cuda")
    print("  architecture:", arch_tag)

    total_train_wall_s = 0.0
    for ep in range(args.epochs):
        _, _, sec = train_one_epoch(
            model,
            criterion,
            optimizer,
            train_ld,
            ep + 1,
            args.epochs,
            device,
            running_loss_every=args.running_loss_every,
        )
        total_train_wall_s += sec

    print()
    print("Total training time (s):", round(total_train_wall_s, 3))

    if args.skip_eval:
        print()
        print("Done (evaluation skipped).")
        return

    test_full = MNIST(root=data_dir, train=False, download=not args.no_download, transform=tfm)
    n_te = trim_to_batches(len(test_full), args.batch_size)
    test_ld = DataLoader(
        Subset(test_full, list(range(n_te))),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    evaluate(model, criterion, test_ld, device, label="Test set (MNIST)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()