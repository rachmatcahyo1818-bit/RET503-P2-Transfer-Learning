
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset_corrected"
METADATA = ROOT / "dataset_raw" / "metadata.csv"
OUTPUT = ROOT / "dataset"

TRAIN_LIGHT = "terang"
VAL_LIGHT = "redup"

if not SOURCE.is_dir():
    sys.exit("ERROR: Folder dataset_corrected tidak ditemukan.")

if not METADATA.is_file():
    sys.exit("ERROR: dataset_raw/metadata.csv tidak ditemukan.")

with open(METADATA, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    required = {"nama_file", "kelas", "kondisi_cahaya"}

    if not required.issubset(set(reader.fieldnames or [])):
        sys.exit("ERROR: Kolom metadata tidak sesuai.")

    rows = list(reader)

if len(rows) != 604:
    sys.exit(f"ERROR: Metadata berisi {len(rows)} baris, bukan 604.")

tasks = []
errors = []

for row in rows:
    filename = (row.get("nama_file") or "").strip()
    kelas = (row.get("kelas") or "").strip()
    cahaya = (row.get("kondisi_cahaya") or "").strip().lower()

    if not filename or not kelas:
        errors.append(f"Nama file/kelas kosong: {row}")
        continue

    if cahaya not in {TRAIN_LIGHT, VAL_LIGHT}:
        errors.append(f"Kondisi cahaya tidak valid: {filename}")
        continue

    source_file = SOURCE / kelas / filename

    if not source_file.is_file():
        errors.append(f"File terkoreksi tidak ditemukan: {source_file}")
        continue

    split = "train" if cahaya == TRAIN_LIGHT else "val"
    tasks.append((source_file, split, kelas, filename))

if errors:
    print("PEMBAGIAN DIBATALKAN. Periksa masalah berikut:")
    for error in errors[:20]:
        print("-", error)
    sys.exit(f"Total masalah: {len(errors)}. Dataset lama tidak diubah.")

if len(tasks) != 604:
    sys.exit("ERROR: Jumlah file yang siap diproses bukan 604.")

# Dataset ini merupakan hasil turunan dan dapat dibuat ulang.
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

counts = Counter()

for source_file, split, kelas, filename in tasks:
    destination = OUTPUT / split / kelas / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, destination)
    counts[(split, kelas)] += 1

print("\nPEMBAGIAN DATA SELESAI")
for kelas in sorted({row["kelas"].strip() for row in rows}):
    train_count = counts[("train", kelas)]
    val_count = counts[("val", kelas)]
    print(f"{kelas}: train={train_count}, val={val_count}")

print(f"\nTotal train: {sum(v for (s, _), v in counts.items() if s == 'train')}")
print(f"Total val:   {sum(v for (s, _), v in counts.items() if s == 'val')}")
print("Train menggunakan kondisi terang.")
print("Validation menggunakan kondisi redup.")
