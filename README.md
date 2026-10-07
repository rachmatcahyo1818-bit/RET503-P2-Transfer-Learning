\# RET503 P2 – Transfer Learning



\## 1. Deskripsi



Proyek ini merupakan implementasi Praktikum P2 mata kuliah RET503 Computer Vision and Deep Learning dengan topik Transfer Learning dan Fine-Tuning Model Visi.



Tujuan praktikum adalah membangun model klasifikasi citra menggunakan dataset proyek sendiri, menerapkan transfer learning, membandingkan tiga strategi pelatihan, serta mengukur performa inferensi model.



Model yang digunakan pada bagian ini adalah \*\*MobileNetV3-Small\*\*.



\---



\## 2. Tujuan



Tujuan eksperimen:



1\. Membangun dataset klasifikasi dari lingkungan proyek.

2\. Menggunakan citra yang telah dikoreksi berdasarkan hasil kalibrasi kamera.

3\. Membagi dataset menjadi training dan validation.

4\. Membandingkan Feature Extraction, Partial Fine-Tuning, dan Scratch Training.

5\. Mengukur akurasi validation dan waktu training.

6\. Mengukur latency inferensi MobileNetV3-Small pada CPU.



\---



\## 3. Dataset



Dataset terdiri dari lima kelas:



| Kelas | Jumlah |

|---|---:|

| arduino\_uno | 99 |

| esp32 | 135 |

| kartu\_rfid | 170 |

| kosong | 50 |

| rfid\_rc522 | 150 |

| \*\*Total\*\* | \*\*604\*\* |



Semua kelas memenuhi ketentuan minimal \*\*50 citra per kelas\*\*.



Metadata dataset disimpan dalam:



```text

dataset\_raw/metadata.csv

