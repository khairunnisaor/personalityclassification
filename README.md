# Laporan Proyek Machine Learning Terapan I - Oryza Khairunnisa
Klasifikasi Kepribadian Introversi vs Extroversi menggunakan Metode Machine Learning Klasik

## Domain Proyek

Kepribadian adalah pola karakteristik pemikiran, perasaan, dan perilaku yang relatif stabil pada individu seiring waktu dan di berbagai situasi. Salah satu dimensi kepribadian yang paling banyak diteliti dan dikenali adalah spektrum introversi-ekstroversi. Individu yang cenderung ekstrovert umumnya digambarkan sebagai pribadi yang ramah, suka bersosialisasi, energik, dan cenderung mencari stimulasi dari luar, seperti interaksi kelompok. Sebaliknya, individu yang cenderung introvert cenderung lebih tenang, reflektif, menyukai waktu sendiri atau interaksi dalam kelompok kecil, serta lebih mudah merasa lelah dengan stimulasi sosial berlebihan. Pemahaman tentang sifat kepribadian seperti introversi dan ekstroversi ini memiliki implikasi signifikan di berbagai bidang, mulai dari pendidikan, pengembangan karier, hingga dinamika interaksi sosial. Secara khusus, di lingkungan pendidikan, mengidentifikasi kecenderungan kepribadian siswa dapat menjadi alat yang sangat berharga untuk personalisasi pembelajaran dan peningkatan hasil akademik.

Meskipun tes kepribadian tradisional (seperti kuesioner) telah lama digunakan, metode ini seringkali memakan banyak waktu, rentan terhadap bias jawaban, dan tidak selalu praktis untuk diterapkan pada skala besar. Berdasarkan hal ini, kebutuhan akan metode identifikasi kepribadian yang lebih efisien dan objektif menjadi krusial. Permasalahan ini dapat diselesaikan dengan memanfaatkan kemajuan dalam bidang machine learning dan kecerdasan buatan.

Proyek ini terinspirasi oleh riset seperti yang dilakukan oleh Sari, Widodo, & Putra (2020) dalam paper mereka "Classification Algorithms to Predict Students' Extraversion-Introversion Traits," mengusulkan penggunaan algoritma klasifikasi untuk memprediksi sifat introversi-ekstroversi. Riset tersebut menunjukkan bahwa dengan menganalisis data perilaku atau karakteristik yang relevan, model machine learning dapat mengidentifikasi pola-pola yang berkaitan dengan sifat kepribadian. Dalam studi mereka, Sari et al. (2020) berhasil menerapkan algoritma klasifikasi seperti Naive Bayes, Decision Tree, dan Support Vector Machine (SVM) untuk memprediksi sifat ekstroversi-introversi siswa dengan akurasi yang menjanjikan, mencapai akurasi hingga 77.2% menggunakan dataset yang berasal dari kuesioner.

Referensi: Sari, N. P., Widodo, T., & Putra, K. A. (2020). Classification Algorithms to Predict Students' Extraversion-Introversion Traits. Journal of Physics: Conference Series, 1566(1), 012061.

## Business Understanding

### Problem Statements
1. Bagaimana cara melakukan identifikasi kepribadian agar lebih efisien, objektif, dan dapat dikembangkan untuk skala yang lebih besar?
2. Algoritma machine learning seperti apa yang dapat digunakan pada data perilaku dan menghasilkan model yang optimal?

### Goals
1. Mengembangkan metode identifikasi kepribadian yang lebih cepat dan bebas dari bias subjektif dibandingkan metode tradisional, juga dapat diterapkan pada jumlah individu yang besar tanpa kehilangan akurasi atau efisiensi.
2. Menemukan metode machine learning terbaik untuk menghasilkan model yang optimal.

### Solution statements
1. Membangun data-driven prediction dengan model machine learning.
   * Menggunakan data perilaku interaksi sosial individu yang kuantitatif, bukan hasil survey kualitatif individu
   * Membangun model machine learning untuk melakukan prediksi kepribadian berdasarkan data perilaku kuantitatif
2. Membandingkan performa dari 3 metode machine learning, berdasarkan kemampuan prediksi dan kecepatan pelatihan model untuk mengukur skalabilitas.
   * Melakukan studi perbandingan metode machine learning untuk mendapatkan model terbaik
   * Melakukan hyperparameter tuning untuk membangun model yang optimal

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

