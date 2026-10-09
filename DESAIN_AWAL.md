# DESAIN AWAL P2 â€” TRANSFER LEARNING

**Mata Kuliah:** RET503 Computer Vision and Deep Learning
**Topik:** Transfer Learning dan Fine-Tuning Model Visi
**Tahap CDIO:** Stage #2 â€” Design

## 1. Misi Proyek

Proyek ini mengembangkan sistem persepsi berbasis kamera untuk mengklasifikasikan komponen elektronik. Sistem dirancang untuk mengenali jenis komponen dan kondisi tanpa objek sebagai prototipe awal klasifikasi visual yang dapat mendukung pengembangan persepsi pada proyek robot.

## 2. Kelas Objek

Dataset terdiri dari lima kelas:

| Kelas | Objek | Jumlah |
|---|---|---:|
| arduino_uno | Board Arduino Uno | 99 |
| esp32 | Board ESP32 | 135 |
| kartu_rfid | Kartu RFID | 170 |
| kosong | Tidak ada objek target | 50 |
| rfid_rc522 | Modul pembaca RFID RC522 | 150 |
| **Total** | | **604** |

Semua kelas memenuhi ketentuan minimal 50 citra per kelas.

## 3. Kamera dan Kalibrasi

Akuisisi data menggunakan webcam laptop dengan resolusi 1280 Ã— 720 piksel. Kalibrasi kamera dilakukan menggunakan 41 foto checkerboard, menghasilkan RMS calibration error sebesar 0,5709 piksel.

Parameter kalibrasi disimpan dalam `calib.npz` dan digunakan dalam proses koreksi distorsi citra. Citra terkoreksi disimpan terpisah dari dataset mentah agar data asli tetap tersedia.

Ketinggian kamera, sudut pemasangan, dan jarak kerja belum didokumentasikan dalam satuan fisik pada dataset awal. Parameter tersebut perlu dicatat ketika kamera dipasang pada dudukan robot sebelum pengujian deployment.

## 4. Unit Komputasi

Pengembangan dan pelatihan dilakukan pada laptop Windows menggunakan Python, PyTorch, dan torchvision. Pelatihan dan pengujian latency saat ini menggunakan CPU tanpa CUDA.

## 5. Target Kinerja

Target referensi dalam materi praktikum adalah 15 FPS atau sekitar 66,7 ms per frame untuk keseluruhan pipeline. Evaluasi mencakup akurasi klasifikasi, waktu pelatihan, dan latency inferensi. Latency inferensi model harus dibedakan dari latency keseluruhan pipeline yang juga mencakup akuisisi kamera, preprocessing, postprocessing, dan komunikasi robot.

## 6. Kandidat Model

Dua kandidat dipertimbangkan berdasarkan materi praktikum:

| Model | Parameter | GFLOPs | Pertimbangan |
|---|---:|---:|---|
| MobileNetV3-Small | â‰ˆ 2,5 juta | â‰ˆ 0,06 | Ringan dan sesuai untuk eksperimen CPU serta perangkat edge |
| EfficientNet-B0 | â‰ˆ 5,3 juta | â‰ˆ 0,39 | Alternatif dengan kebutuhan komputasi lebih besar |

MobileNetV3-Small dipilih sebagai model utama karena kebutuhan komputasinya relatif rendah. EfficientNet-B0 menjadi alternatif untuk evaluasi lanjutan. Nilai parameter dan GFLOPs merupakan perkiraan referensi dari materi, bukan hasil pengukuran pada dataset ini.

## 7. Strategi Transfer Learning

Eksperimen membandingkan tiga pendekatan menggunakan arsitektur MobileNetV3-Small:

1. **Feature Extraction:** backbone pretrained ImageNet dibekukan dan classifier baru dilatih.
2. **Partial Fine-Tuning:** dua blok feature terakhir dan classifier dilatih, sedangkan bagian backbone lainnya dibekukan.
3. **Scratch Training:** seluruh model dilatih dari bobot awal tanpa pretrained weights.

Input model berukuran 224 Ã— 224 piksel dengan normalisasi statistik ImageNet. Augmentasi training mencakup random resized crop, horizontal flip, dan color jitter. Ketiga mode dilatih selama 10 epoch untuk membandingkan akurasi validation dan waktu pelatihan.

## 8. Rencana dan Pembagian Data

Metadata disimpan pada `dataset_raw/metadata.csv` dengan informasi nama file, kelas, tanggal, dan kondisi cahaya. Dataset mentah dipertahankan, sedangkan citra hasil koreksi dan pembagian data disimpan di folder terpisah.

| Kelas | Training: Terang | Validation: Redup |
|---|---:|---:|
| arduino_uno | 49 | 50 |
| esp32 | 65 | 70 |
| kartu_rfid | 88 | 82 |
| kosong | 25 | 25 |
| rfid_rc522 | 70 | 80 |
| **Total** | **297** | **307** |

Pembagian berdasarkan kondisi cahaya digunakan untuk menguji kemampuan generalisasi terhadap perubahan pencahayaan. Hasil validation perlu ditafsirkan sebagai evaluasi lintas kondisi cahaya, bukan sebagai estimasi kinerja universal pada semua lingkungan.

## 9. Risiko dan Mitigasi

| Risiko | Mitigasi |
|---|---|
| Overfitting pada data training | Menggunakan pretrained weights, augmentasi, dan evaluasi validation |
| Perubahan kondisi cahaya | Menguji pada kondisi cahaya yang berbeda dan menambah variasi data |
| Data leakage | Memisahkan data berdasarkan kondisi atau sesi pengambilan, bukan hanya random split |
| Ketidakseimbangan kelas | Memantau jumlah citra setiap kelas dan menambah data bila diperlukan |
| Latency meningkat pada robot | Mengukur latency seluruh pipeline pada perangkat target sebelum deployment |

**Status desain:** Model utama, dataset, strategi eksperimen, dan rencana evaluasi telah ditetapkan. Parameter fisik dudukan kamera dan pengujian kinerja pada perangkat robot menjadi pekerjaan lanjutan.
