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
1. Pengecekan dan Pengisian Nilai yang Hilang (missing values)
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
[429 rows x 8 columns]
```
Berdasarkan pengecekan di atas, diketahui bahwa terdapat 429 baris yang terduplikat. Oleh karena itu, dilakukan penghapusan baris yang redundan tersebut.

```python
df_personality = df_personality.drop_duplicates()
```

3. Pengecekan Outlier
```python
numeric_features = df_personality[df_personality.columns].select_dtypes(include=['number']).columns

# Mengidentifikasi outliers menggunakan IQR
Q1 = df_personality[numeric_features].quantile(0.25)
Q3 = df_personality[numeric_features].quantile(0.75)
IQR = Q3 - Q1

# Menghapus Outlier sesuai perhitungan Kuartil
# Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik
condition = ~((df_personality[numeric_features] < (Q1 - 1.5*IQR)) | (df_personality[numeric_features] > (Q3 + 1.5*IQR))).any(axis=1)
df_filtered_numeric = df_personality.loc[condition, numeric_features]

# Menggabungkan kembali dengan kolom kategorikal
categorical_features = df_personality.select_dtypes(include=['object']).columns
df_personality = pd.concat([df_filtered_numeric, df_personality.loc[condition, categorical_features]], axis=1)
```

4. Menampilkan informasi rangkuman data secara umum.
```python
df_personality.info()
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
Index: 2471 entries, 0 to 2899
Data columns (total 8 columns):
 #   Column                     Non-Null Count  Dtype 
---  ------                     --------------  ----- 
 0   Time_spent_Alone           2471 non-null   int64 
 1   Stage_fear                 2471 non-null   int64 
 2   Social_event_attendance    2471 non-null   int64 
 3   Going_outside              2471 non-null   int64 
 4   Drained_after_socializing  2471 non-null   int64 
 5   Friends_circle_size        2471 non-null   int64 
 6   Post_frequency             2471 non-null   int64 
 7   Personality                2471 non-null   object
dtypes: int64(7), object(1)
memory usage: 173.7+ KB
```
Dari informasi keseluruhan di atas dapat diketahui jumlah baris data setelah penghapusan data duplikat dan outlier. Dari 2.900 baris, data bersih tersisa 2.471 dan sudah tidak ada kolom yang memiliki missing values karena telah dilakukan imputation. Dari informasi ini juga diketahui bahwa selutuh fitur independen telah memiliki tipe data integer, dimana tidak diperlukan lagi konversi kategorikal data menjadi numerikal di tahap data preparation.

### Exploratory Data Analysis
Untuk mengetahui persebaran dan behavior dari data, dilakukan beberapa hal yaitu:

1. Analisis Distribusi
![alt text](https://github.com/khairunnisaor/personalityclassification/blob/main/images/distribution_updated.png)
Diagram batang di atas menggambarkan distribusi data pada setiap kolom. Hampir seluruh distribusi variabel memiliki kecenderungan miring ke kanan (right-skewed), dimana lebih banyak nilai mendekati minimal. Sedangkan, saat dibandingkan frekuensi variabel target yang ditunjukkan pada grafik di bawah, hasilnya adalah lebih banyak data dengan label extrovert, walaupun ketidakseimbangannya tidak terlalu signifikan.

![alt text](https://github.com/khairunnisaor/personalityclassification/blob/main/images/count_target.png)

3. Analisis Bivariate
<br>Untuk mengetahui hubungan setiap fitur dengan variabel target, dilakukan analisis bivariate dimana setiap fitur dikelompokkan berdasarkan nilai variabel targetnya dan dirata-rata.
![alt text](https://github.com/khairunnisaor/personalityclassification/blob/main/images/bivariate.png)

Berdasarkan diagram batang di atas, dapat diketahui bahwa variabel `time spent alone`, `stage fear`, dan `drained after socializing` berkaitan erat dengan kepribadian Introvert. Sedangkan sisa variabel lainnya berkaitan erat dengan kepribadian extrovert.

Hal ini juga terlihat jelas pada correlation analisis di bawah. Terdapat tiga nilai yang dapat diketahui dari correlation table, yaitu:
* Nilai negatif: Semakin nilai antar dua variabel mendekati -1, artinya kedua variabel tersebut berkorelasi negatif atau terbalik.
* Nilai positif: Semakin nilai antar dua variabel mendekati +1, artinya kedua variabel tersebut berkorelasi positif atau searah.
* Nilai nol: Jika nilai antar dua variabel cenderung mendekati nol, artinya kedua variabel tersebut saling tidak berkorelasi.

![alt text](https://github.com/khairunnisaor/personalityclassification/blob/main/images/corr.png)

Pada tabel korelasi ini, dapat dilihat bahwa `time spent alone`, `stage fear`, dan `drained after socializing` memiliki korelasi yang positif satu sama lain, namun berkorelasi sangat negatif dengan variabel sisanya.



## Data Preparation
Setelah memahami data yang akan digunakan untuk melatih model machine learning dengan baik, selanjutnya adalah data preparation. Pada tahap ini dilakukan transformasi dan pembagian data agar sesuai dengan input yang dibutuhkan untuk proses training. Beberapa tahapan yang dilakukan yaitu:

1. Standarisasi atau Normalisasi Fitur
<br>Pada tahapan ini, dilakukan penyeragaman atau standarisasi skala variabel independen agar seluruh fitur memiliki nilai minimal dan nilai maksimal yang sama, sehingga tidak ada data yang terlalu tinggi atau terlalu rendah nilainya. Tahapan ini adalah langkah yang krusial dalam membangun model machine learning, karena tanpa fitur yang terstandarisasi, model akan susah mempelajari kesamaan pola yang ada dalam data.

```python
# Memastikan hanya data dengan tipe numerikal yang akan diproses
numeric_features = df_data.select_dtypes(include=['number']).columns
numeric_features

# Standardisasi fitur numerik
scaler = StandardScaler()
df_data[numeric_features] = scaler.fit_transform(df_data[numeric_features])
```

2. Data Splitting
<br>Setelah memastikan bahwa fitur independen berada dalam skala yang serupa, dilakukan langkah terakhir sebelum melatih model machine learning, yaitu data splitting. Tahapan ini dilakukan untuk membagi data menjadi dua, yaitu train dan test set. Train set akan digunakan untuk tahapan pelatihan model, sedangkan test set akan digunakan untuk evaluasi model. Hal ini dilakukan agar model yang dihasilkan objektif, dimana model harus menghasilkan prediksi yang akurat dan tidak mengetahui "jawaban" dari data test. Pada tahap ini, dipilih 80% dari keseluruhan data menjadi data pelatihan dan 20% sisanya digunakan untuk evaluasi.

```python
# Pisahkan fitur independen (X) dan target (y)
X = df_data.drop(columns=['Personality'])
y = df_data['Personality']

# Encode target (y), mengubah data target dari kategorikal atau format string menjadi numerik
y = LabelEncoder().fit_transform(y)

# Split data menjadi set pelatihan dan set uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tampilkan bentuk set pelatihan dan set uji untuk memastikan split
print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
```
Output:
```
Training set shape: X_train=(1976, 7), y_train=(1976,)
Test set shape: X_test=(495, 7), y_test=(495,)
```



## Modeling
Tujuan utama dari tahap modeling adalah mengoptimalkan kinerja model agar mampu membuat prediksi atau keputusan yang akurat pada data baru yang belum pernah dilihat sebelumnya. Optimalisasi ini umumnya dicapai dengan meminimalkan loss function, yang mengukur seberapa jauh prediksi model menyimpang dari nilai atau label aktual dalam data pelatihan. Selama iterasi pelatihan, model terus-menerus menyesuaikan parameternya berdasarkan gradien fungsi kerugian, bergerak menuju konfigurasi yang paling efisien dalam memprediksi output yang diinginkan.

Pada tahap ini, dibandingkan tiga algoritma machine learning klasik untuk mendapatkan model yang paling baik dan optimal yaitu K-Nearest Neighbor (KNN), Support Vector Machine (SVM), dan Naive Bayes.

1. KNN
<br>K-Nearest Neighbors (KNN) adalah salah satu algoritma machine learning yang paling sederhana, non-parametrik, dan lazy learning (pembelajar malas) yang utamanya digunakan untuk klasifikasi dan juga bisa untuk regresi. KNN digolongkan sebagai lazy learning karena tidak membangun model secara eksplisit selama fase pelatihan; semua komputasi terjadi ketika ada data baru yang perlu diklasifikasikan atau diprediksi.

2. SVM


3. Naive Bayes


### Hyperparameter Tuning

----------
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

