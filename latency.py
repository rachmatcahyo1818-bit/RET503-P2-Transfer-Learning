import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


MODEL_PATH = Path("models/mobilenet_v3_small_feature.pth")
IMAGE_SIZE = 224
WARMUP = 20
RUNS = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

classes = checkpoint["classes"]
num_classes = len(classes)

# Buat model
model = mobilenet_v3_small(weights=None)

in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    num_classes
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Input dummy 1 frame
dummy = torch.randn(
    1, 3, IMAGE_SIZE, IMAGE_SIZE,
    device=DEVICE
)

print("=" * 60)
print("LATENCY TEST - MobileNetV3-Small")
print("=" * 60)
print("Model :", MODEL_PATH)
print("Device:", DEVICE)
print("Input :", f"1 x 3 x {IMAGE_SIZE} x {IMAGE_SIZE}")
print("Kelas :", classes)

# Warm-up
with torch.inference_mode():
    for _ in range(WARMUP):
        _ = model(dummy)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

# Measurement
times = []

with torch.inference_mode():
    for _ in range(RUNS):

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(dummy)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        times.append((end - start) * 1000.0)

times_sorted = sorted(times)

avg_ms = sum(times) / len(times)
median_ms = times_sorted[len(times_sorted) // 2]

p95_index = int(0.95 * len(times_sorted)) - 1
p95_ms = times_sorted[max(0, p95_index)]

min_ms = min(times)
max_ms = max(times)

fps = 1000.0 / avg_ms

print()
print("HASIL:")
print(f"Min latency    : {min_ms:.3f} ms")
print(f"Average latency: {avg_ms:.3f} ms")
print(f"Median latency : {median_ms:.3f} ms")
print(f"P95 latency    : {p95_ms:.3f} ms")
print(f"Max latency    : {max_ms:.3f} ms")
print(f"Estimated FPS  : {fps:.2f}")

print()
print("TARGET REFERENSI PPT:")
print("15 FPS = sekitar 66.7 ms/frame")
print("=" * 60)