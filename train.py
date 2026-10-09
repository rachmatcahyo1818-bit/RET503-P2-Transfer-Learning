"""RET503 P2 - MobileNetV3-Small training.

Runs Feature Extraction, Partial Fine-Tuning, and Scratch Training.
Expected folders:
  dataset/train/<class_name>/...
  dataset/val/<class_name>/...

For the current experiment, train should contain bright images and val should
contain dim/redup images. Existing outputs in results/ and models/ are replaced
with results from this run.
"""

import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

BATCH_SIZE = 16
EPOCHS = 10
SEED = 42
IMAGE_SIZE = 224
EXPECTED_CLASSES = ["arduino_uno", "esp32", "kartu_rfid", "kosong", "rfid_rc522"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK DATASET FOLDERS
# ============================================================

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"

if not train_dir.is_dir() or not val_dir.is_dir():
    raise SystemExit(
        "ERROR: Folder dataset/train dan dataset/val tidak ditemukan. "
        "Jalankan split.py terlebih dahulu."
    )


# ============================================================
# PREPROCESSING: 224x224, RGB, IMAGENET NORMALIZATION
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASET AND DATALOADERS
# ============================================================

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

if train_dataset.classes != val_dataset.classes:
    raise SystemExit(
        "ERROR: Daftar kelas train dan validation berbeda. "
        f"Train={train_dataset.classes}; val={val_dataset.classes}"
    )

if train_dataset.classes != EXPECTED_CLASSES:
    raise SystemExit(
        "ERROR: Nama kelas tidak sesuai dengan dataset P2. "
        f"Ditemukan: {train_dataset.classes}; diharapkan: {EXPECTED_CLASSES}"
    )

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise SystemExit("ERROR: Dataset train atau validation kosong.")

NUM_CLASSES = len(train_dataset.classes)

loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=loader_generator,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Device:", DEVICE)
print("Classes:", train_dataset.classes)
print("Class-to-index:", train_dataset.class_to_idx)
print("Jumlah kelas:", NUM_CLASSES)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Split protocol: training set / validation set dibuat dari split.py")
print("Batch size:", BATCH_SIZE, "| Epochs:", EPOCHS, "| Seed:", SEED)
print("Catatan: train.py akan melatih ketiga mode dan memperbarui hasil sebelumnya.")


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(mode: str) -> nn.Module:
    """Create MobileNetV3-Small with mode-specific trainable layers."""
    if mode in ("feature", "partial"):
        weights = MobileNet_V3_Small_Weights.DEFAULT
    elif mode == "scratch":
        weights = None
    else:
        raise ValueError("Mode harus: feature, partial, atau scratch")

    model = mobilenet_v3_small(weights=weights)

    # Replace final ImageNet classifier output for the project's 5 classes.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    if mode == "scratch":
        # All parameters begin from random initialization.
        for parameter in model.parameters():
            parameter.requires_grad = True

    else:
        # Freeze the entire network first.
        for parameter in model.parameters():
            parameter.requires_grad = False

        # Feature extraction trains the complete classifier head.
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

        if mode == "partial":
            # Partial fine-tuning: unfreeze the last two feature modules
            # plus the complete classifier head.
            for block in list(model.features)[-2:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    return model.to(DEVICE)


# ============================================================
# KEEP FROZEN BATCHNORM STATISTICS FIXED
# ============================================================

def keep_frozen_batchnorm_in_eval(model: nn.Module, mode: str) -> None:
    """After model.train(), keep BatchNorm in frozen layers in eval mode.

    This prevents running means/variances in frozen pretrained layers from
    being updated by small batches. Trainable BatchNorm modules in the opened
    partial-tuning feature blocks remain in training mode.
    """
    if mode not in ("feature", "partial"):
        return

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            has_trainable_affine_params = any(
                parameter.requires_grad
                for parameter in module.parameters(recurse=False)
            )
            if not has_trainable_affine_params:
                module.eval()


# ============================================================
# VALIDATION
# ============================================================

def validate(model: nn.Module) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_n = labels.size(0)
            total += batch_n
            correct += (predictions == labels).sum().item()
            loss_sum += loss.item() * batch_n

    return 100.0 * correct / total, loss_sum / total


# ============================================================
# TRAIN ONE MODE
# ============================================================

def train_mode(mode: str) -> dict:
    print("\n" + "=" * 64)
    print("TRAINING MODE:", mode.upper())
    print("=" * 64)

    model = create_model(mode)

    if mode == "feature":
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )
    elif mode == "partial":
        feature_params = []
        classifier_params = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("classifier"):
                classifier_params.append(parameter)
            else:
                feature_params.append(parameter)

        optimizer = optim.Adam([
            {"params": feature_params, "lr": 1e-4},
            {"params": classifier_params, "lr": 1e-3},
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    epoch_90 = None
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        keep_frozen_batchnorm_in_eval(model, mode)

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_n = labels.size(0)
            running_loss += loss.item() * batch_n
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += batch_n

        scheduler.step()

        train_loss = running_loss / train_total
        train_accuracy = 100.0 * train_correct / train_total
        val_accuracy, val_loss = validate(model)

        history.append({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_accuracy": f"{train_accuracy:.4f}",
            "val_loss": f"{val_loss:.6f}",
            "val_accuracy": f"{val_accuracy:.4f}",
        })

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            checkpoint = {
                "architecture": "mobilenet_v3_small",
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "mode": mode,
                "best_val_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "input_size": IMAGE_SIZE,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "split_protocol": "dataset/train and dataset/val; see metadata.csv and split.py",
            }
            torch.save(checkpoint, MODEL_DIR / f"mobilenet_v3_small_{mode}.pth")

        if epoch_90 is None and val_accuracy >= 90.0:
            epoch_90 = epoch

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | "
            f"Train {train_accuracy:.2f}% | "
            f"Val Loss {val_loss:.4f} | "
            f"Val {val_accuracy:.2f}%"
        )

    elapsed = time.perf_counter() - start_time

    history_file = RESULT_DIR / f"history_{mode}.csv"
    with history_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch", "train_loss", "train_accuracy",
                "val_loss", "val_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nSelesai:", mode)
    print(f"Best Val Accuracy : {best_accuracy:.2f}%")
    print(f"Best Epoch        : {best_epoch}")
    print(f"Epoch >= 90%      : {epoch_90 if epoch_90 is not None else 'belum tercapai'}")
    print(f"Training Time     : {elapsed:.2f} detik")

    return {
        "mode": mode,
        "best_val_accuracy": f"{best_accuracy:.2f}",
        "best_epoch": best_epoch,
        "epoch_90": epoch_90 if epoch_90 is not None else "-",
        "training_time_sec": f"{elapsed:.2f}",
    }


# ============================================================
# RUN ALL THREE MODES
# ============================================================

all_results = []
for training_mode in ("feature", "partial", "scratch"):
    all_results.append(train_mode(training_mode))

summary_file = RESULT_DIR / "training_summary.csv"
with summary_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "mode", "best_val_accuracy", "best_epoch",
            "epoch_90", "training_time_sec",
        ],
    )
    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# PLOT VALIDATION ACCURACY PER EPOCH
# ============================================================

plt.figure(figsize=(9, 6))
for training_mode in ("feature", "partial", "scratch"):
    history_file = RESULT_DIR / f"history_{training_mode}.csv"
    epochs = []
    val_accuracies = []

    with history_file.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            epochs.append(int(row["epoch"]))
            val_accuracies.append(float(row["val_accuracy"]))

    plt.plot(epochs, val_accuracies, marker="o", label=training_mode)

plt.axhline(90.0, linestyle="--", linewidth=1, label="90% reference")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("MobileNetV3-Small - Validation Accuracy")
plt.xticks(range(1, EPOCHS + 1))
plt.ylim(0, 105)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_file = RESULT_DIR / "validation_accuracy.png"
plt.savefig(plot_file, dpi=200)
plt.close()

print("\n" + "=" * 64)
print("SEMUA TRAINING SELESAI")
print("=" * 64)
print("Ringkasan:", summary_file)
print("Grafik:", plot_file)
print("Model:", MODEL_DIR)
print("Mode yang dijalankan: feature, partial, scratch")
