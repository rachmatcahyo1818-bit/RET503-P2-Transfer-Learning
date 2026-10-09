# RET503 P2 — Transfer Learning untuk Klasifikasi Komponen Elektronik

## 1. Ringkasan proyek

Proyek P2 membandingkan tiga strategi training **MobileNetV3-Small** untuk mengklasifikasikan lima kelas citra: `arduino_uno`, `esp32`, `kartu_rfid`, `kosong`, dan `rfid_rc522`. Hasil yang dicatat mencakup akurasi validation, waktu training, epoch pertama saat akurasi validation mencapai 90%, dan latency inferensi pada CPU.

## 2. Dataset

Dataset mentah berisi **604 gambar**. Pengambilan citra dikonfirmasi berlangsung pada **6 Oktober 2026**; seluruh baris `dataset_raw/metadata.csv` telah menggunakan tanggal `2026-10-06`.

| Kelas | Gambar mentah | Train (terang) | Validation (redup) |
|---|---:|---:|---:|
| `arduino_uno` | 99 | 49 | 50 |
| `esp32` | 135 | 65 | 70 |
| `kartu_rfid` | 170 | 88 | 82 |
| `kosong` | 50 | 25 | 25 |
| `rfid_rc522` | 150 | 70 | 80 |
| **Total** | **604** | **297** | **307** |

Pemisahan dilakukan menurut kondisi cahaya: citra **terang** digunakan untuk training dan citra **redup** untuk validation. Ini menjadi uji lintas kondisi pencahayaan, tetapi hasilnya tidak otomatis menjamin performa pada semua lingkungan atau foto yang benar-benar baru.

Kolom metadata: `nama_file`, `kelas`, `tanggal`, dan `kondisi_cahaya`.

## 3. Preprocessing dan konfigurasi

- Ukuran input: `224 × 224` piksel, RGB.
- Normalisasi ImageNet: mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`.
- Augmentasi training: `RandomResizedCrop(224)`, `RandomHorizontalFlip`, dan `ColorJitter`.
- Validation: `Resize(256)` lalu `CenterCrop(224)`.
- Batch size: 16; epoch: 10; seed: 42.
- Loss: Cross Entropy; optimizer: Adam; scheduler: Cosine Annealing.
- Perangkat training dan latency: CPU; CUDA tidak digunakan.

### Strategi training

1. **Feature Extraction** — backbone MobileNetV3-Small pretrained ImageNet dibekukan; classifier baru dilatih.
2. **Partial Fine-Tuning** — dua modul feature terakhir dan classifier dilatih; backbone lainnya dibekukan.
3. **Scratch Training** — semua parameter dilatih dari bobot awal tanpa pretrained weights.

## 4. Hasil training

| Metode | Best validation accuracy | Best epoch | Epoch pertama dengan validation accuracy ≥90% | Waktu training |
|---|---:|---:|---:|---:|
| Feature Extraction | **100.00%** | 1 | 1 | 136.32 detik |
| Partial Fine-Tuning | **100.00%** | 2 | 1 | 142.28 detik |
| Scratch Training | 26.71% | 1 | Tidak tercapai | 194.86 detik |

Feature Extraction dan Partial Fine-Tuning mencapai 100% pada validation set eksperimen ini. Feature Extraction mencapai hasil terbaik pada epoch pertama dan memiliki waktu training lebih singkat sehingga checkpoint tersebut dipilih untuk pengukuran latency.

### Validation accuracy per epoch

| Epoch | Feature Extraction | Partial Fine-Tuning | Scratch Training |
|---:|---:|---:|---:|
| 1 | 100.00% | 94.14% | 26.71% |
| 2 | 100.00% | 100.00% | 26.71% |
| 3 | 99.35% | 100.00% | 26.71% |
| 4 | 100.00% | 100.00% | 26.71% |
| 5 | 100.00% | 98.70% | 26.71% |
| 6 | 100.00% | 100.00% | 26.71% |
| 7 | 100.00% | 100.00% | 26.71% |
| 8 | 100.00% | 100.00% | 26.71% |
| 9 | 100.00% | 99.02% | 26.71% |
| 10 | 100.00% | 98.70% | 26.71% |

![Validation Accuracy per Epoch](results/validation_accuracy.png)

### Analisis singkat

Scratch Training memperoleh validation accuracy 26.71%, sama dengan proporsi kelas terbesar (`kartu_rfid`, 82 dari 307 gambar validation). Ini mungkin menunjukkan kecenderungan memprediksi kelas mayoritas; confusion matrix, precision, recall, dan F1-score per kelas belum dihitung dan diperlukan untuk memastikannya.

Akurasi 100% adalah hasil pada validation set yang digunakan, bukan jaminan akurasi pada semua foto baru. Evaluasi terpisah pada gambar dari sesi pengambilan berbeda tetap disarankan untuk memeriksa generalisasi dan kemungkinan kemiripan data.

## 5. Hasil latency

Pengukuran `latency.py` menggunakan checkpoint Feature Extraction, CPU, dan satu input `1 × 3 × 224 × 224`.

| Metrik | Hasil |
|---|---:|
| Minimum latency | 12.010 ms |
| Average latency | **14.869 ms** |
| Median latency | 14.906 ms |
| P95 latency | 16.895 ms |
| Maximum latency | 27.206 ms |
| Estimated FPS | **67.25 FPS** |

Materi praktikum menetapkan referensi 15 FPS atau sekitar 66,7 ms per frame untuk **keseluruhan pipeline**. Hasil 14.869 ms di sini hanya mengukur inferensi model; tidak termasuk akuisisi kamera, preprocessing, postprocessing, komunikasi, atau ROS 2. Karena itu, 67.25 FPS tidak diklaim sebagai FPS keseluruhan sistem.

## 6. File utama

- `dataset_raw/metadata.csv` — metadata dataset.
- `calib.npz` dan `correct_images.py` — berkas kalibrasi dan koreksi distorsi.
- `split.py` — membentuk dataset train/validation berdasarkan kondisi cahaya.
- `train.py` — menjalankan ketiga mode training.
- `models/mobilenet_v3_small_feature.pth` — checkpoint Feature Extraction.
- `models/mobilenet_v3_small_partial.pth` — checkpoint Partial Fine-Tuning.
- `models/mobilenet_v3_small_scratch.pth` — checkpoint Scratch Training.
- `results/history_feature.csv`, `results/history_partial.csv`, `results/history_scratch.csv` — riwayat per epoch.
- `results/training_summary.csv` — ringkasan ketiga metode.
- `results/validation_accuracy.png` — grafik validation accuracy.
- `results/latency_summary.csv` — ringkasan latency.

## 7. Menjalankan ulang eksperimen

Dari folder utama proyek setelah dependensi tersedia:

```powershell
py -m pip install -r requirements.txt
py correct_images.py
py split.py
py train.py
py latency.py
```

Perintah di atas adalah untuk mengulang eksperimen dan akan memperbarui hasil pada `models/` dan `results/`. **Tidak perlu dijalankan untuk sekadar membaca atau mengumpulkan hasil eksperimen yang sudah ada.** Pastikan `calib.npz`, `dataset_raw/`, dan metadata tersedia sebelum preprocessing.

## 8. Keterbatasan dan pekerjaan lanjutan

- Hitung confusion matrix, precision, recall, dan F1-score per kelas, khususnya untuk Scratch Training.
- Uji pada gambar baru dengan variasi jarak, orientasi, latar, dan cahaya.
- Ukur latency seluruh pipeline bila memakai kamera langsung atau ROS 2.
- Tinggi, sudut, dan jarak kerja webcam belum tercatat dalam pengukuran awal; catat nilai aktual sebelum deployment pada robot.
