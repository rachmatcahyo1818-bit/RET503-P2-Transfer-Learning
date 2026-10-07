\# DESAIN AWAL P2 – TRANSFER LEARNING



\*\*Mata Kuliah:\*\* RET503 Computer Vision and Deep Learning  

\*\*Pertemuan:\*\* 3 – Transfer Learning dan Fine-Tuning Model Visi  

\*\*CDIO Stage:\*\* #2 Design  



\## 1. Misi Proyek



Sistem persepsi digunakan untuk mengenali komponen elektronik menggunakan kamera sebagai bagian dari pipeline visi pada proyek robot. Model klasifikasi digunakan untuk mengenali objek yang berada di depan kamera dan menjadi dasar pengambilan informasi visual untuk sistem robot.



\## 2. Kelas Objek



Dataset terdiri dari lima kelas objek:



| Kelas | Keterangan |

|---|---|

| `arduino\_uno` | Board Arduino Uno |

| `esp32` | Board ESP32 |

| `kartu\_rfid` | Kartu RFID |

| `kosong` | Kondisi tanpa objek |

| `rfid\_rc522` | Modul pembaca RFID RC522 |



\## 3. Kamera dan Dudukan



Kamera yang digunakan adalah webcam laptop dengan resolusi \*\*1280 × 720 piksel\*\*. Kamera digunakan untuk memperoleh citra objek pada area pengamatan. Parameter kalibrasi kamera diperoleh pada Pertemuan 2 dan digunakan untuk melakukan koreksi distorsi sebelum citra digunakan pada proses pembelajaran.



\## 4. Unit Komputasi



Proses pengembangan dan pelatihan model dilakukan pada laptop menggunakan \*\*CPU\*\*. PyTorch dan torchvision digunakan sebagai framework pembelajaran mendalam. Sistem saat ini menggunakan pemrosesan CPU tanpa CUDA.



\## 5. Target Kinerja



Target sistem adalah memperoleh akurasi validasi minimal \*\*90%\*\* serta waktu inferensi yang cukup rendah untuk mendukung aplikasi robot. Materi menggunakan target referensi \*\*15 FPS atau sekitar 66,7 ms/frame\*\* untuk keseluruhan pipeline.



\## 6. Kandidat Model



Model yang dipilih adalah \*\*MobileNetV3-Small\*\*. Model ini dipilih karena memiliki ukuran dan kebutuhan komputasi yang relatif ringan sehingga sesuai untuk penggunaan pada perangkat edge dan aplikasi robot. Berdasarkan materi, MobileNetV3-Small memiliki sekitar \*\*2,5 juta parameter\*\* dan sekitar \*\*0,06 GFLOPs\*\*.



\## 7. Strategi Transfer Learning



Tiga pendekatan dibandingkan:



1\. \*\*Feature Extraction\*\*  

&#x20;  Backbone pretrained dibekukan dan hanya classifier yang dilatih.



2\. \*\*Partial Fine-Tuning\*\*  

&#x20;  Bagian akhir backbone dan classifier dilatih menggunakan learning rate yang lebih kecil pada backbone.



3\. \*\*Scratch Training\*\*  

&#x20;  Seluruh model dilatih dari bobot awal tanpa pretrained weights.



Preprocessing citra menggunakan ukuran input \*\*224 × 224 piksel\*\*, format RGB, dan normalisasi berdasarkan statistik ImageNet.



\## 8. Rencana Data



Dataset terdiri dari \*\*604 citra\*\* dari lima kelas.



| Kelas | Jumlah Citra |

|---|---:|

| `arduino\_uno` | 99 |

| `esp32` | 135 |

| `kartu\_rfid` | 170 |

| `kosong` | 50 |

| `rfid\_rc522` | 150 |

| \*\*Total\*\* | \*\*604\*\* |



Dataset kemudian diproses menggunakan parameter kalibrasi kamera untuk menghasilkan citra terkoreksi. Dataset dibagi menjadi:



\- \*\*Training:\*\* 483 citra

\- \*\*Validation:\*\* 121 citra



Metadata dataset disimpan dalam `metadata.csv`.



\## 9. Risiko dan Mitigasi



| Risiko | Mitigasi |

|---|---|

| Overfitting | Menggunakan transfer learning dan augmentasi data |

| Variasi pencahayaan | Menyediakan variasi kondisi pencahayaan pada dataset |

| Data leakage | Memisahkan data training dan validation secara hati-hati |

| Inferensi terlalu lambat | Menggunakan model ringan dan melakukan pengukuran latency |

| Ketidakseimbangan jumlah data | Memantau jumlah citra tiap kelas dan menambah data bila diperlukan |



\### Hasil Awal Eksperimen



Eksperimen awal MobileNetV3-Small menghasilkan:



| Mode | Best Validation Accuracy |

|---|---:|

| Feature Extraction | \*\*100,00%\*\* |

| Partial Fine-Tuning | \*\*100,00%\*\* |

| Scratch | \*\*28,10%\*\* |



Pengujian latency model MobileNetV3-Small pada CPU menghasilkan:



\- Average latency: \*\*14,911 ms\*\*

\- Estimated FPS: \*\*67,06 FPS\*\*



Hasil tersebut akan dianalisis lebih lanjut pada README dan laporan akhir P2.

