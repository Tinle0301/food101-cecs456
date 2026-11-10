"""Shared code for both notebooks: classes, data split, data loaders,
training loop and evaluation. Both models use exactly the same data and metrics.
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"              # Food-101 is downloaded here (not committed)
SPLIT_FILE = ROOT / "split.json"      # the shared train/val/test split
RESULTS_DIR = ROOT / "results"
SEED = 42

# 20 Food-101 classes, including look-alike groups on purpose:
# steak / filet_mignon / prime_rib, ramen / pho,
# chocolate_cake / chocolate_mousse, spaghetti_bolognese / spaghetti_carbonara
CLASSES = [
    "apple_pie", "caesar_salad", "chocolate_cake", "chocolate_mousse", "dumplings",
    "filet_mignon", "french_fries", "fried_rice", "hamburger", "ice_cream",
    "pancakes", "pho", "pizza", "prime_rib", "ramen",
    "spaghetti_bolognese", "spaghetti_carbonara", "steak", "sushi", "tacos",
]
VAL_PER_CLASS = 150                    # 750 official train images -> 600 train + 150 val
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet normalization


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- data split
def _find_food101(root):
    for meta in Path(root).rglob("meta/train.json"):
        return meta.parent.parent
    return None


def prepare_split(data_dir=DATA_DIR, download=True):
    """Download Food-101 (~5 GB) if needed and write split.json.

    Train/val come from the official train list (600/150 per class, seed 42);
    test is the official test list (250 per class). Each image is in exactly one split.
    """
    base = _find_food101(data_dir)
    if base is None:
        if not download:
            raise FileNotFoundError(f"Food-101 not found under {data_dir}")
        from torchvision.datasets import Food101
        print("Downloading Food-101 (~5 GB)...")
        Food101(root=str(data_dir), split="train", download=True)
        base = _find_food101(data_dir)

    train_meta = json.loads((base / "meta" / "train.json").read_text())
    test_meta = json.loads((base / "meta" / "test.json").read_text())
    rng = random.Random(SEED)
    split = {"classes": CLASSES, "base_dir": str(base.resolve()), "train": [], "val": [], "test": []}
    for label, cls in enumerate(CLASSES):
        ids = sorted(train_meta[cls])
        rng.shuffle(ids)
        split["val"] += [[f"images/{i}.jpg", label] for i in ids[:VAL_PER_CLASS]]
        split["train"] += [[f"images/{i}.jpg", label] for i in ids[VAL_PER_CLASS:]]
        split["test"] += [[f"images/{i}.jpg", label] for i in sorted(test_meta[cls])]

    paths = {k: {p for p, _ in split[k]} for k in ("train", "val", "test")}
    assert not (paths["train"] & paths["val"] or paths["train"] & paths["test"] or paths["val"] & paths["test"])
    SPLIT_FILE.write_text(json.dumps(split))
    for k in ("train", "val", "test"):
        print(f"{k:5s}: {len(split[k])} images")
    return split


class FoodDataset(Dataset):
    def __init__(self, items, base_dir, transform):
        self.items, self.base_dir, self.transform = items, Path(base_dir), transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(self.base_dir / path).convert("RGB")), label


def get_loaders(img_size, batch_size, num_workers=2, limit=0):
    """Train loader uses augmentation; val/test use a fixed resize + center crop.
    `limit` > 0 keeps only that many train/val images (for a quick test run)."""
    if not SPLIT_FILE.exists():
        prepare_split()
    split = json.loads(SPLIT_FILE.read_text())
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    items = {k: split[k] for k in ("train", "val", "test")}
    if limit:
        rng = random.Random(SEED)
        items["train"] = rng.sample(items["train"], min(limit, len(items["train"])))
        items["val"] = rng.sample(items["val"], min(limit, len(items["val"])))
    ds = {"train": FoodDataset(items["train"], split["base_dir"], train_tf),
          "val": FoodDataset(items["val"], split["base_dir"], eval_tf),
          "test": FoodDataset(items["test"], split["base_dir"], eval_tf)}
    return {k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"), num_workers=num_workers,
                          pin_memory=torch.cuda.is_available()) for k, v in ds.items()}


# ---------------------------------------------------------------- training
def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given. Returns (loss, accuracy, macro-F1)."""
    train = optimizer is not None
    model.train(train)
    total, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            y_true += y.tolist()
            y_pred += logits.argmax(1).tolist()
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return total / len(y_true), acc, f1_score(y_true, y_pred, average="macro")


def train_model(model, loaders, optimizer, epochs, name, scheduler=None, on_epoch_start=None):
    """Train and keep the weights with the best validation macro-F1.

    on_epoch_start(epoch) may return a new (optimizer, scheduler), e.g. to unfreeze layers.
    Saves results/<name>/best.pt and history.json. Returns the history dict.
    """
    device = get_device()
    model.to(device)
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    loss_fn = nn.CrossEntropyLoss()
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    best, start = -1.0, time.time()
    for epoch in range(1, epochs + 1):
        if on_epoch_start:
            new = on_epoch_start(epoch)
            if new:
                optimizer, scheduler = new
        tr = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        va = run_epoch(model, loaders["val"], loss_fn, device)
        if scheduler:
            scheduler.step()
        for k, v in zip(["train_loss", "train_acc"], tr[:2]):
            hist[k].append(v)
        for k, v in zip(["val_loss", "val_acc", "val_f1"], va):
            hist[k].append(v)
        print(f"Epoch {epoch:2d}/{epochs} | train loss {tr[0]:.3f} acc {tr[1]:.3f} | "
              f"val loss {va[0]:.3f} acc {va[1]:.3f} macro-F1 {va[2]:.3f}")
        if va[2] > best:
            best = va[2]
            hist["best_epoch"] = epoch
            torch.save(model.state_dict(), out / "best.pt")
    hist["train_minutes"] = round((time.time() - start) / 60, 1)
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"Best val macro-F1 {best:.3f} at epoch {hist['best_epoch']} ({hist['train_minutes']} min)")
    return hist


def plot_curves(hist, title):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="Train")
    ax[0].plot(ep, hist["val_loss"], label="Valid")
    ax[0].set(title=f"Loss for each epoch ({title})", xlabel="Epoch", ylabel="Loss")
    ax[1].plot(ep, hist["train_acc"], label="Train")
    ax[1].plot(ep, hist["val_acc"], label="Valid")
    ax[1].set(title=f"Accuracy for each epoch ({title})", xlabel="Epoch", ylabel="Accuracy")
    for a in ax:
        a.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loaders, name):
    """Test-set metrics with the best weights. Saves results/<name>/test_metrics.json."""
    device = get_device()
    model.load_state_dict(torch.load(RESULTS_DIR / name / "best.pt", map_location=device))
    model.to(device).eval()
    probs, y_true = [], []
    for X, y in loaders["test"]:
        probs.append(torch.softmax(model(X.to(device)), 1).cpu())
        y_true += y.tolist()
    probs = torch.cat(probs).numpy()
    y_true = np.array(y_true)
    y_pred = probs.argmax(1)
    losses = -np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    confused = Counter((CLASSES[t], CLASSES[q]) for t, q in zip(y_true, y_pred) if t != q)
    metrics = {"loss": float(losses.mean()), "accuracy": float((y_true == y_pred).mean()),
               "precision": float(p), "recall": float(r), "f1": float(f1),
               "top_confused": [[t, q, c] for (t, q), c in confused.most_common(10)]}
    (RESULTS_DIR / name / "test_metrics.json").write_text(json.dumps(metrics, indent=1))
    print("Test set: " + " | ".join(f"{k} {metrics[k]:.4f}" for k in
                                    ["loss", "accuracy", "precision", "recall", "f1"]))
    print("Most confused (actual -> predicted):", metrics["top_confused"][:5])
    return {"y_true": y_true, "y_pred": y_pred, "probs": probs, "losses": losses, "metrics": metrics}


def plot_confusion(res, title):
    cm = confusion_matrix(res["y_true"], res["y_pred"], labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix ({title})")
    fig.tight_layout()
    return fig


def plot_top_losses(res, loaders, title, n=9):
    """The misclassified test images with the greatest losses: Prediction / Actual / Loss / Probability."""
    ds = loaders["test"].dataset
    wrong = [i for i in np.argsort(-res["losses"]) if res["y_true"][i] != res["y_pred"][i]][:n]
    fig, axes = plt.subplots(3, 3, figsize=(10, 12))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        img = ds[i][0].permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
        ax.imshow(np.clip(img, 0, 1))
        pred, act = res["y_pred"][i], res["y_true"][i]
        ax.set_title(f"{CLASSES[pred]} / {CLASSES[act]}\n"
                     f"{res['losses'][i]:.2f} / {res['probs'][i, pred]:.2f}", fontsize=9)
    fig.suptitle(f"Misclassified examples with the greatest losses ({title})\n"
                 "Prediction / Actual / Loss / Probability")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------- saving
def download_results(name):
    """Zip results/<name> (curves, metrics; weights excluded) and download it.
    On Colab this saves the zip to your computer, since Colab deletes files when the session ends."""
    import zipfile
    out = ROOT / f"results_{name}.zip"
    with zipfile.ZipFile(out, "w") as z:
        for f in (RESULTS_DIR / name).iterdir():
            if f.suffix != ".pt":
                z.write(f, f"results/{name}/{f.name}")
    print(f"Saved {out.name}: unzip it into the repo folder, then git add / commit / push.")
    try:
        from google.colab import files
        files.download(str(out))
    except ImportError:
        pass
    return out
