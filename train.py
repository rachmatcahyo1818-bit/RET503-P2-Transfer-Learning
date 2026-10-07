import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ============================================================
# KONFIGURASI
# ============================================================

DATA_DIR = Path("dataset")
RESULT_DIR = Path("results")
MODEL_DIR = Path("models")

BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

RESULT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# PREPROCESSING SESUAI PPT
# 224x224, RGB, ImageNet mean/std
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

NUM_CLASSES = len(train_dataset.classes)

print("Classes:", train_dataset.classes)
print("Jumlah kelas:", NUM_CLASSES)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))


# ============================================================
# MODEL
# ============================================================

def create_model(mode):

    if mode in ("feature", "partial"):
        weights = MobileNet_V3_Small_Weights.DEFAULT
    else:
        weights = None

    model = mobilenet_v3_small(weights=weights)

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # Semua backbone frozen, hanya classifier dilatih
    # --------------------------------------------------------
    if mode == "feature":

        for p in model.parameters():
            p.requires_grad = False

    # --------------------------------------------------------
    # PARTIAL FINE-TUNING
    # Backbone frozen, buka 2 blok terakhir
    # dan classifier
    # --------------------------------------------------------
    elif mode == "partial":

        for p in model.parameters():
            p.requires_grad = False

        # Buka dua blok feature terakhir
        for block in list(model.features)[-2:]:
            for p in block.parameters():
                p.requires_grad = True

    # --------------------------------------------------------
    # SCRATCH
    # Semua layer trainable dan tanpa pretrained weights
    # --------------------------------------------------------
    elif mode == "scratch":

        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError("Mode harus: feature, partial, atau scratch")

    # Ganti classifier terakhir
    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    return model


# ============================================================
# FUNGSI VALIDASI
# ============================================================

def validate(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100.0 * correct / total

    return accuracy


# ============================================================
# TRAINING SATU MODE
# ============================================================

def train_mode(mode):

    print("\n" + "=" * 60)
    print("TRAINING MODE:", mode.upper())
    print("=" * 60)

    model = create_model(mode)

    # --------------------------------------------------------
    # Optimizer sesuai konsep PPT
    # --------------------------------------------------------

    if mode == "feature":

        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3
        )

    elif mode == "partial":

        backbone_params = []
        classifier_params = []

        for name, p in model.named_parameters():

            if not p.requires_grad:
                continue

            if name.startswith("classifier"):
                classifier_params.append(p)
            else:
                backbone_params.append(p)

        optimizer = optim.Adam([
            {
                "params": backbone_params,
                "lr": 1e-4
            },
            {
                "params": classifier_params,
                "lr": 1e-3
            }
        ])

    else:  # scratch

        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    criterion = nn.CrossEntropyLoss()

    history = []

    best_accuracy = 0.0
    best_epoch = 0
    epoch_90 = None

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        scheduler.step()

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_accuracy = validate(model)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "mode": mode,
                    "best_val_accuracy": best_accuracy
                },
                MODEL_DIR / f"mobilenet_v3_small_{mode}.pth"
            )

        if epoch_90 is None and val_accuracy >= 90.0:
            epoch_90 = epoch

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | "
            f"Train {train_accuracy:.2f}% | "
            f"Val {val_accuracy:.2f}%"
        )

    elapsed = time.perf_counter() - start_time

    # --------------------------------------------------------
    # SIMPAN HISTORY
    # --------------------------------------------------------

    history_file = RESULT_DIR / f"history_{mode}.csv"

    with open(
        history_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print("\nSelesai:", mode)
    print(f"Best Val Accuracy : {best_accuracy:.2f}%")
    print(f"Best Epoch        : {best_epoch}")

    if epoch_90 is not None:
        print(f"Epoch >= 90%      : {epoch_90}")
    else:
        print("Epoch >= 90%      : belum tercapai")

    print(f"Training Time     : {elapsed:.2f} detik")

    return {
        "mode": mode,
        "best_val_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "epoch_90": epoch_90 if epoch_90 is not None else "-",
        "training_time_sec": elapsed,
        "history": history
    }


# ============================================================
# JALANKAN 3 MODE
# ============================================================

all_results = []

for mode in ["feature", "partial", "scratch"]:

    result = train_mode(mode)

    all_results.append({
        "mode": result["mode"],
        "best_val_accuracy": f"{result['best_val_accuracy']:.2f}",
        "best_epoch": result["best_epoch"],
        "epoch_90": result["epoch_90"],
        "training_time_sec": f"{result['training_time_sec']:.2f}"
    })


# ============================================================
# SIMPAN HASIL UTAMA
# ============================================================

summary_file = RESULT_DIR / "training_summary.csv"

with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "mode",
            "best_val_accuracy",
            "best_epoch",
            "epoch_90",
            "training_time_sec"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# GRAFIK VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(9, 6))

for mode in ["feature", "partial", "scratch"]:

    history_file = RESULT_DIR / f"history_{mode}.csv"

    epochs = []
    val_acc = []

    with open(
        history_file,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            epochs.append(int(row["epoch"]))
            val_acc.append(float(row["val_accuracy"]))

    plt.plot(
        epochs,
        val_acc,
        marker="o",
        label=mode
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("MobileNetV3-Small - Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "validation_accuracy.png",
    dpi=200
)

plt.close()

print("\n" + "=" * 60)
print("SEMUA TRAINING SELESAI")
print("=" * 60)
print("Hasil:", summary_file)
print("Grafik:", RESULT_DIR / "validation_accuracy.png")
print("Model:", MODEL_DIR)