# RET503 P2 — Transfer Learning untuk Klasifikasi Komponen Elektronik

## 1. Ringkasan proyek

Proyek ini membandingkan tiga strategi training **MobileNetV3-Small** untuk mengklasifikasikan lima kategori pada citra: `arduino_uno`, `esp32`, `kartu_rfid`, `kosong`, dan `rfid_rc522`. Evaluasi mencakup akurasi validasi, waktu training, epoch pertama saat akurasi validasi mencapai 90%, dan latency inferensi pada CPU.

## 2. Dataset

Dataset mentah berisi **604 gambar**.

| Kelas | Gambar mentah | Train (terang) | Validation (redup) |
|---|---:|---:|---:|
| `arduino_uno` | 99 | 49 | 50 |
| `esp32` | 135 | 65 | 70 |
| `kartu_rfid` | 170 | 88 | 82 |
| `kosong` | 50 | 25 | 25 |
| `rfid_rc522` | 150 | 70 | 80 |
| **Total** | **604** | **297** | **307** |

Data dipisahkan berdasarkan kondisi pencahayaan pada metadata: citra **terang** digunakan untuk training dan citra **redup** untuk validation. Dengan demikian, validation menguji kondisi cahaya yang berbeda dari training. Angka ini perlu ditafsirkan dalam konteks pembagian tersebut dan bukan dianggap sebagai jaminan performa pada semua kondisi nyata.

Kolom metadata: `nama_file`, `kelas`, `tanggal`, dan `kondisi_cahaya`.

## 3. Preprocessing dan konfigurasi

- Ukuran input: `224 × 224` piksel, RGB.
- Normalisasi ImageNet: mean `[0.485, 0.456, 0.406]` dan standard deviation `[0.229, 0.224, 0.225]`.
- Augmentasi training: `RandomResizedCrop(224)`, `RandomHorizontalFlip`, dan `ColorJitter`.
- Validation: `Resize(256)` lalu `CenterCrop(224)`.
- Batch size: 16; jumlah epoch: 10; seed: 42.
- Loss: Cross Entropy; optimizer: Adam; scheduler: Cosine Annealing.
- Perangkat training dan pengujian latency: CPU.

### Strategi training

1. **Feature Extraction** — backbone MobileNetV3-Small menggunakan bobot pralatih ImageNet dan dibekukan; classifier baru dilatih.
2. **Partial Fine-Tuning** — dua blok feature terakhir dan classifier dilatih; layer backbone lainnya dibekukan.
3. **Scratch Training** — model dimulai tanpa bobot pralatih dan seluruh parameter dilatih.

## 4. Hasil training terbaru

| Metode | Best validation accuracy | Best epoch | Epoch pertama dengan validation accuracy ≥90% | Waktu training |
|---|---:|---:|---:|---:|
| Feature Extraction | **100.00%** | 1 | 1 | 136.32 detik |
| Partial Fine-Tuning | **100.00%** | 2 | 1 | 142.28 detik |
| Scratch Training | 26.71% | 1 | Tidak tercapai | 194.86 detik |

Feature Extraction dan Partial Fine-Tuning sama-sama mencapai 100% pada validation set dalam eksperimen ini. Feature Extraction mencapai nilai terbaik lebih awal dan waktu training-nya lebih singkat. Karena itu, checkpoint Feature Extraction dipilih untuk pengujian latency.

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

Grafik validation accuracy dibuat otomatis oleh program training:

![Validation Accuracy per Epoch](results/validation_accuracy.png)

### Catatan interpretasi

Scratch Training menghasilkan validation accuracy 26.71%, sama dengan proporsi kelas terbesar pada validation set (`kartu_rfid`, 82 dari 307 gambar). Ini mengindikasikan model mungkin cenderung memprediksi kelas mayoritas; confusion matrix serta precision, recall, dan F1-score per kelas perlu diperiksa untuk memastikan perilakunya.

Akurasi 100% hanya menggambarkan validation set pada eksperimen ini. Pengujian terpisah dengan gambar baru dari sesi pengambilan berbeda tetap diperlukan sebelum menyimpulkan generalisasi model. Pembagian berdasarkan kondisi pencahayaan merupakan aspek penting dalam menafsirkan hasil ini.

## 5. Hasil latency

Pengukuran dilakukan menggunakan `latency.py` dengan checkpoint Feature Extraction, CPU, dan satu input berukuran `1 × 3 × 224 × 224`.

| Metrik | Hasil |
|---|---:|
| Minimum latency | 12.010 ms |
| Average latency | **14.869 ms** |
| Median latency | 14.906 ms |
| P95 latency | 16.895 ms |
| Maximum latency | 27.206 ms |
| Estimated FPS | **67.25 FPS** |

Target referensi pada materi adalah 15 FPS, setara sekitar 66.7 ms per frame. Rata-rata latency inferensi model 14.869 ms berada di bawah angka tersebut. Pengukuran ini **hanya mencakup inferensi model**, bukan keseluruhan pipeline kamera, pengolahan citra, post-processing, komunikasi, atau ROS 2; jadi 67.25 FPS bukan hasil pengukuran FPS seluruh sistem.

## 6. File hasil

- `models/mobilenet_v3_small_feature.pth` — checkpoint Feature Extraction terbaik.
- `models/mobilenet_v3_small_partial.pth` — checkpoint Partial Fine-Tuning terbaik.
- `models/mobilenet_v3_small_scratch.pth` — checkpoint Scratch Training terbaik.
- `results/history_feature.csv` — riwayat Feature Extraction per epoch.
- `results/history_partial.csv` — riwayat Partial Fine-Tuning per epoch.
- `results/history_scratch.csv` — riwayat Scratch Training per epoch.
- `results/training_summary.csv` — ringkasan ketiga metode.
- `results/validation_accuracy.png` — grafik validation accuracy.
- `results/latency_summary.csv` — ringkasan latency.

## 7. Menjalankan ulang eksperimen

Jalankan dari folder utama proyek setelah Python dan dependensi tersedia:

```powershell
py -m pip install -r requirements.txt
py correct_images.py
py split.py
py train.py
py latency.py
```

`correct_images.py` menyiapkan citra hasil koreksi dengan berkas kalibrasi proyek. `split.py` membentuk `dataset/train` dan `dataset/val` berdasarkan metadata. `train.py` melatih ketiga strategi dan menghasilkan checkpoint, CSV, serta grafik. `latency.py` mengukur inferensi checkpoint Feature Extraction.

Pastikan `calib.npz`, `dataset_raw/`, dan `dataset_raw/metadata.csv` tersedia sebelum menjalankan preprocessing.

## 8. Keterbatasan dan pekerjaan lanjutan

- Evaluasi dengan confusion matrix, precision, recall, dan F1-score per kelas, khususnya untuk Scratch Training.
- Uji model pada gambar baru yang belum digunakan selama pengembangan, dengan variasi jarak, orientasi, latar, dan pencahayaan.
- Ukur FPS pipeline secara menyeluruh bila menggunakan kamera langsung atau ROS 2.
- Lengkapi dokumentasi posisi kamera (tinggi, sudut, dan jarak kerja aktual) berdasarkan pengukuran perangkat yang benar-benar digunakan.
