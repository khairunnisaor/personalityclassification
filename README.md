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
2. Menemukan metode machine learning terbaik dari segi performa prediksi dan waktu pelatihan.
   
### Solution statements
1. Membangun data-driven prediction dengan model machine learning.
   * Menggunakan data perilaku interaksi sosial individu yang kuantitatif, bukan hasil survey kualitatif individu
   * Membangun model machine learning untuk melakukan prediksi kepribadian berdasarkan data perilaku kuantitatif
2. Membandingkan performa dari 3 metode machine learning, berdasarkan kemampuan prediksi dan kecepatan pelatihan model untuk mengukur skalabilitas.
   * Melakukan studi perbandingan metode machine learning untuk mendapatkan model terbaik
   * Melakukan hyperparameter tuning untuk membangun model yang optimal

## Data Understanding
Dataset ini berisi tentang data perilaku seseorang yang menggambarkan apakah individu tersebut memiliki kepribadian yang introvert atau extrovert. Berdasarkan data perilaku invidu yang kuantitatif, dataset ini dapat menjadi sumber daya yang berharga untuk psikolog dan peneliti dalam mempelajari dan mengeksplorasi bagaimana perilaku sosial dapat dieksplorasi untuk mengidentifikasi spektrum kepribadian manusia.

Dataset ini terdiri dari 7 jenis kebiasaan dari 2.900 responden tentang kondisi mental dan cara mereka berinteraksi sosial, yang diukur dalam skala yang telah ditentukan.
Sumber dataset: [Extrovert vs. Introvert Behavior Data] (https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data)

Beberapa kegunaan dataset: Membuat model machine learning untuk prediksi kepribadian, menganalisa korelasi antara perilaku sosial dan kepribadian, membuat visualisasi tentang tren kebiasaan atau perilaku sosial individu, dan lain-lain.

### Variabel-variabel pada Extrovert vs. Introvert Behavior Data adalah sebagai berikut:
Kondisi Psikis
- Stage_fear: Memiliki demam panggung (Yes/No).
- Drained_after_socializing: Perasaan lelah setelah bersosialisasi (Yes/No).

Interaksi Sosial
- Time_spent_Alone: Jumlah jam yang dihabiskan sendirian dalam satu hari (0–11).
- Social_event_attendance: Frekuensi menghadiri acara sosial (0–10).
- Going_outside: Frekuensi bepergian keluar (0–7).
- Friends_circle_size: Jumlah teman dekat (0–15).
- Post_frequency: Frekuensi mengunggah sesuatu ke media sosial (0–10).

Variabel Target
- Personality: Identifikasi kepribadian (Extrovert/Introvert).

### Mengecek Kondisi dan Eksplorasi Data
1. Pengecekan dan Pengisian Nilai yang Hilang
```python
# Memeriksa jumlah nilai yang hilang di setiap kolom
missing_values = df_personality.isnull().sum()
missing_values[missing_values > 0]
```
| Variabel        | Jumlah Nilai Hilang |
| --------------- |:-------------------:|
| Time_spent_Alone | 63 |
| Stage_fear | 73 |
| Social_event_attendance | 62 |
| Going_outside | 66 |
| Drained_after_socializing | 52 |
| Friends_circle_size | 77 |
| Post_frequency | 65 |

```python
# Membuat fungsi pengisian nilai hilang menggunakan KNNImputer, untuk mengisi data hilang berdasarkan tetangga terdekatnya
def imputeMissingValues(df, col):
    imputer = KNNImputer(n_neighbors=2, weights="distance")
    fill_na = imputer.fit_transform(df[col].values.reshape(-1,1)).astype(np.int64)
    return fill_na

# Mengubah variabel dengan tipe data kategori (Yes/No) menjadi angka (1/0) agar dapat dilakukan imputasi
df_personality[['Stage_fear','Drained_after_socializing']] = df_personality[['Stage_fear','Drained_after_socializing']].apply(
                                                                  lambda series: pd.Series(
                                                                        LabelEncoder().fit_transform(series[series.notnull()]),
                                                                        index=series[series.notnull()].index)
                                                               )
# Melakukan imputasi untuk mengisi nilai yang hilang
df_personality['Time_spent_Alone'] = imputeMissingValues(df_personality, 'Time_spent_Alone')
df_personality['Social_event_attendance'] = imputeMissingValues(df_personality, 'Social_event_attendance')
df_personality['Going_outside'] = imputeMissingValues(df_personality, 'Going_outside')
df_personality['Post_frequency'] = imputeMissingValues(df_personality, 'Post_frequency')
df_personality['Friends_circle_size'] = imputeMissingValues(df_personality, 'Friends_circle_size')

df_personality['Stage_fear'] = imputeMissingValues(df_personality, 'Stage_fear')
df_personality['Drained_after_socializing'] = imputeMissingValues(df_personality, 'Drained_after_socializing')
```
2. Pengecekan Data Duplikat
```python
# Mengidentifikasi baris duplikat
duplicates = df_personality.duplicated()

print("Baris duplikat:")
print(df_personality[duplicates])
```
Output:
```
Baris duplikat:
      Time_spent_Alone  Stage_fear  Social_event_attendance  Going_outside  Drained_after_socializing  \
47                  10           1                        1              2                          1
217                  5           1                        2              0                          1
...                ...         ...                      ...            ...                        ...
2892                 9           1                        2              0                          1
2895                 3           0                        7              6                          0

      Drained_after_socializing  Friends_circle_size  Post_frequency  Personality
47                            1                    2               0    Introvert    
217                           1                    2               0    Introvert     
...                         ...                  ...             ...          ...    
2892                          1                    1               2    Introvert   
2895                          0                    6               6    Extrovert   
```

3. dfad



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

