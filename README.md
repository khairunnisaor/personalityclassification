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
Sumber dataset: [Extrovert vs. Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data)

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


### Exploratory Data Analysis
Untuk mengetahui informasi tentang statistik dataset secara umum, distribusi, dan karakteristik dari data, dilakukan beberapa hal yaitu:

1. Menampilkan informasi dataset
```python
df_personality.info()
```

Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2900 entries, 0 to 2899
Data columns (total 8 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Time_spent_Alone           2837 non-null   float64
 1   Stage_fear                 2827 non-null   object 
 2   Social_event_attendance    2838 non-null   float64
 3   Going_outside              2834 non-null   float64
 4   Drained_after_socializing  2848 non-null   object 
 5   Friends_circle_size        2823 non-null   float64
 6   Post_frequency             2835 non-null   float64
 7   Personality                2900 non-null   object 
dtypes: float64(5), object(3)
memory usage: 181.4+ KB
```
Secara umum, diketahui bahwa terdapat 2.900 baris data dengan hampir semua kolom memiliki missing value. Selain variabel target `Personality`, terdapat dua kolom yang bertipe kategorikal: `Stage_fear` dan `Drained_after_socializing`. Hal ini mengindikasikan perlu dilakukannya missing value handling dan categorical data encoding pada tahap data preparation.

2. Analisis Distribusi
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
Setelah memahami data yang akan digunakan untuk melatih model machine learning dengan baik, selanjutnya adalah data preparation. Pada tahap ini dilakukan penanganan nilai yang hilang, penghapusan data yang terduplikat, dan pengecekan Outlier. Setelah penanganan untuk menghasilkan data yang bersih dan siap digunakan ini selesai, dilanjutkan dengan tahap transformasi dan pembagian data agar sesuai dengan input yang dibutuhkan untuk proses training. Beberapa tahapan yang dilakukan yaitu:

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
Dari informasi keseluruhan di atas dapat diketahui jumlah baris data setelah penghapusan data duplikat dan outlier. Dari 2.900 baris, data bersih tersisa 2.471 dan sudah tidak ada kolom yang memiliki missing values karena telah dilakukan imputation. Dari informasi ini juga diketahui bahwa selutuh fitur independen telah memiliki tipe data integer, dimana tidak diperlukan lagi konversi kategorikal data menjadi numerikal.

Setelah data sudah bersih dan lengkap, dilanjutkan dengan dua tahapan, yaitu standarisasi fitur dan data splitting agar dapat diproses dengan model machine learning dengan mudah.

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

Pada tahap ini, dibandingkan tiga algoritma machine learning klasik untuk mendapatkan model yang paling baik dan optimal yaitu K-Nearest Neighbor (KNN), Support Vector Machine (SVM), dan Naive Bayes. Parameter yang digunakan pada pembangunan model baseline adalah parameter default, yang kemudian akan dioptimalisasi pada tahap hyperparameter tuning.

1. KNN
<br>K-Nearest Neighbors adalah salah satu algoritma machine learning yang paling sederhana, non-parametrik, dan lazy learning (pembelajar malas) yang utamanya digunakan untuk klasifikasi dan juga bisa untuk regresi. KNN digolongkan sebagai lazy learning karena tidak membangun model secara eksplisit selama fase pelatihan; semua komputasi terjadi ketika ada data baru yang perlu diklasifikasikan atau diprediksi.

2. SVM
<br>Support Vector Machine adalah algoritma machine learning yang kuat dan serbaguna, utamanya digunakan untuk tugas klasifikasi dan juga bisa untuk regresi. SVM bekerja dengan menemukan hyperplane (bidang pemisah) optimal dalam ruang berdimensi tinggi yang secara jelas memisahkan titik-titik data dari berbagai kelas. Ide inti di balik SVM adalah menemukan "batas" terbaik yang memisahkan kelas-kelas dalam data.

3. Naive Bayes
<br>Naive Bayes adalah algoritma klasifikasi probabilistik yang didasarkan pada Teorema Bayes dengan asumsi "naif" (sederhana) tentang independensi antar fitur. Meskipun asumsi ini seringkali tidak sepenuhnya realistis di dunia nyata, Naive Bayes seringkali menunjukkan kinerja yang sangat baik dalam berbagai aplikasi klasifikasi, terutama dalam klasifikasi teks. Inti dari algoritma Naive Bayes adalah Teorema Bayes, yang secara matematis menggambarkan probabilitas suatu peristiwa, berdasarkan pengetahuan sebelumnya tentang kondisi yang mungkin terkait dengan peristiwa tersebut.

Pembangunan model baseline dilakukan sebagai berikut.
```python
# Bagian 1: Pelatihan Model
knn = KNeighborsClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)
nb = BernoulliNB().fit(X_train, y_train)

print("Model training selesai.")
```

Kelebihan dan kekurangan dari ketiga algoritma di atas adalah sebagai berikut:
| Algoritma        | Kelebihan | Kekurangan |
| ---------------- | --------- | ---------- |
| KNN | Sederhana dan mudah dipahami, fleksibel untuk berbagai distribusi data, mudah dalam penambahan data baru | Biaya komputasi tinggi saat prediksi, sensitif terhadap noise dan outlier, kurang efektif pada data dimensi tinggi  |
| SVM | Efektif pada ruang berdimensi tinggi, memiliki generalisasi yang baik, efisiensi memori | Membutuhkan waktu pelatihan yang lama pada dataset besar, sensitif terhadap pemilihan kernel dan hyperparameter, sulit diinterpretasikan |
| Naive Bayes | Sederhana dan mudah diimplementasikan, sangat cepat dalam pelatihan dan prediksi, dapat menangani data kategorikan dan numerik | Asumsi fitur independen yang 'naif', estimasi probalitias yang tidak akurat, tidak bisa belajar interaksi antar fitur |


### Hyperparameter Tuning
Untuk mendapatkan model dengan performa terbaik, dilakukan hyperparameter tuning. Tahapan ini adalah langkah yang dilakukan untuk mendapatkan kombinasi parameter terbaik untuk menghasilkan model yang paling optimal. Pada tahap ini, dilakukan proses pencarian hyperparameter dengan menggunakan Grid Search.

```python
# Membuat fungsi untuk melakukan model tuning
def model_tuning(X_train, y_train, X_test, y_test, estimator, param_grid):
    # Inisialisasi GridSearchCV
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Output hasil terbaik
    print(f"Best parameters (Grid Search): {grid_search.best_params_}")
    best_grid = grid_search.best_estimator_

    # Evaluasi performa model pada test set
    grid_search_score = best_grid.score(X_test, y_test)
    print(f"Accuracy after Grid Search: {grid_search_score:.4f}")

    return best_grid, round(grid_search_score, 4)
```

Fungsi di atas kemudian digunakan untuk melakukan model tuning pada ketiga algoritma. Berikut ditampilkan contoh penggunaan fungsi tuning pada algoritma KNN.
```python
# Definisikan parameter grid untuk Grid Search (contoh untuk KNN)
param_grid_knn = {'n_neighbors': [1,10, 1],
                  'leaf_size': [20,40,1],
                  'p': [1,2],
                  'weights': ['uniform', 'distance'],
                  'metric': ['minkowski', 'chebyshev']}

best_param_knn, best_score_knn = model_tuning(X_train, y_train, X_test, y_test, knn, param_grid_knn)
tuning_summary.append(["KNN", best_score_knn])
```

Hyperparameter terbaik yang didapatkan dari eksperimen menggunakan Grid Search untuk setiap algoritma adalah sebagai berikut:
1. KNN
* leaf_size (ambang batas jumlah sampel dalam daun pohon pencarian): 20
* metric (perhitungan jarak yang digunakan): minkowski
* n_neighbors (jumlah tetangga terdekat yang diperhitungkan): 10
* p (power parameter untuk minkowski): 2
* weights (pembobotan untuk setiap tetangga terdekat): uniform

2. SVM
* C (toleransi model terhadap misclassification): 0.001
* gamma (spengaruh satu sampel data pelatihan tunggal terhadap batas keputusan): 1000
* kernel (fungsi untuk memetakan data input ke ruang dimensi yang lebih tinggi): sigmoid

3. Naive Bayes
* alpha (nilai Laplace/Lidstone smoothing): 0.001
* binarize (ambang batas konversi fitur numerik): 0.0
* fit_prior (penggunaan probabilitas prior kelas): True

Untuk tahapan pemilihan model terbaik, akan dilakukan pada tahap selanjutnya yaitu evaluasi.

## Evaluation
Pada tahap ini dilakukan evaluasi terhadap model yang telah dibuat. Evaluasi dilakukan dengan membandingkan performa antar model, juga sebelum dan sesudah hyperparameter tuning, berdasarkan beberapa metrik seperti akurasi, precision, recall, F1 score, dan waktu pelatihan.

Untuk memahami metrik-metrik ini, penting untuk mengenal empat istilah dasar dari confusion matrix (matriks kebingungan):

Untuk dapat menghitung metrik akurasi, precision, recall, dan F1 score, terdapat empat istilah yang digunakan dalam mengevaluasi hasil prediksi model yang biasa ditunjukkan dalam bentuk confusion matrix. Keempat istilah tersebut adalah:
* True Positive (TP): Jumlah kasus positif yang diprediksi dengan benar sebagai positif
* True Negative (TN): Jumlah kasus negatif yang diprediksi dengan benar sebagai negatif
* False Positive (FP): Jumlah kasus negatif yang salah diprediksi sebagai positif (Error Tipe I)
* False Negative (FN): Jumlah kasus positif yang salah diprediksi sebagai negatif (Error Tipe II)

Berdasarkan keempat nilai di atas, akurasi, precision, recall, dan F1 score dihitung dengan cara sebagai berikut:
1. Akurasi
<br>Akurasi mengukur proporsi total prediksi yang benar dari semua prediksi yang dibuat. Ini adalah metrik paling intuitif dan memberikan gambaran umum seberapa sering model membuat prediksi yang tepat.
<br>**Formula:** Accuracy = (TP + TN) / (TP + TN + FP + FN)

2. Presisi
<br>Presisi mengukur proporsi positif yang benar dari semua kasus yang diprediksi sebagai positif. Dengan kata lain, dari semua yang diprediksi sebagai kelas positif, berapa banyak yang benar-benar positif. Ini menjawab pertanyaan: "Seberapa banyak prediksi positif saya yang memang benar?"
<br>**Formula:** Precision = TP / (TP + FP)

3. Recall (Sensitivitas / Tingkat Keberhasilan)
<br>Recall mengukur proporsi positif yang teridentifikasi dengan benar dari semua kasus positif aktual. Dengan kata lain, dari semua kasus yang seharusnya positif, berapa banyak yang berhasil ditemukan oleh model. Ini menjawab pertanyaan: "Seberapa banyak kasus positif yang sebenarnya berhasil ditemukan oleh model saya?"
<br>**Formula:** Recall = TP / (TP + FN)

4. F1-Score
<br>F1-Score adalah rata-rata harmonik dari Presisi dan Recall. Ini memberikan skor tunggal yang menyeimbangkan Presisi dan Recall. F1-Score tinggi menunjukkan bahwa model memiliki Presisi dan Recall yang baik secara bersamaan.
<br>**Formula:** F1_Score = 2 ∗ (Precision ∗ Recall) / (Precision + Recall)

Selain keempat metrik di atas, dilakukan juga perhitungan training time untuk mengukur skalabilitas model pada data yang lebih besar dan algoritma yang lebih kompleks.
5. Training Time
<br>Training time adalah waktu yang dibutuhkan model untuk melakukan pelatihan. 
<br>**Formula:** Training time = start training time - end training time

Berdasarkan metrik di atas, didapatkan hasil eksperimen baseline model dengan parameter default sebagai berikut:
| Model        | Akurasi | Precision | Recall | F1-Score |
| ------------ | ------- | --------- | ------ | -------- |
| KNN          | 91.11   | 91.37     | 91.11  | 91.14    | 
| SVM          | 92.53   | 92.80     | 92.53  | 92.55    |
| NaiveBayes   | 92.32   | 92.57     | 92.32  | 92.35    |

Lalu, dilakukan hyperparameter tuning dan hasil eksperimen dengan best parameter beserta waktu pelatihannya sebagai berikut:
| Model             | Akurasi | Precision | Recall | F1-Score | Training Time (s) |
| ----------------- | ------- | --------- | ------ | -------- | ----------------- |
| Tuned-KNN         | 91.72   | 92.00     | 91.72  | 91.75    | 0.004474          | 
| Tuned-SVM         | 92.53   | 92.80     | 92.53  | 92.55    | 0.083815          | 
| Tuned-NaiveBayes  | 92.32   | 92.57     | 92.32  | 92.35    | 0.003291          | 


Berdasarkan hasil eksperimen dalam membandingkan tiga algoritma machine learning klasik dan memanfaatkan hyperparameter tuning untuk memperoleh model optimal, didapatkan hasil yaitu:
* Berdasarkan F1-Score dan Akurasi, model terbaik adalah model dengan algoritma SVM, dimana model ini sudah memiliki performa yang cukup baik sejak awal. Hal ini tercermin dari F1-Score sebesar 92.55 dengan baseline model dan tidak ada peningkatan yang signifikan setelah dilakukan hyperparameter tuning.
* Hal yang sama juga terjadi pada model dengan Naive Bayes, dimana tidak ada peningkatan setelah dilakukan hyperparameter tuning. Hal ini mungkin terjadi karena parameter grid yang kurang luas atau metode best parameter search yang belum optimal. Kemungkinan lain adalah dibutuhkannya tahap preprocessing yang lebih baik, contohnya opsi ketika memperlakukan missing values dan melakukan data balancing untuk masing-masing kelas.
* Algoritma KNN memiliki performa di bawah dua algoritma yang lain, namun terjadi sedikit peningkatan (+0.61) pada F1-Score setelah hyperparameter tuning.
* Untuk menjawab skalabilitas proyek, diketahui bahwa model dengan performa paling cepat pada pelatihan adalah Naive Bayes, dengan waktu 0.0033s. Walaupun performa model dengan Naive Bayes berada pada peringkat kedua di bawah SVM, jelas bahwa bayes hampir 30 kali lebih cepat dari SVM. Dengan selisih waktu yang signifikan namun performa yang hanya berselisih 0.2 poin, dapat direkomendasikan bahwa ketika data jauh lebih besar (saat ini terdapat 2.4K baris data), Naive Bayes adalah win-win solution untuk kedua aspek.
* KNN memiliki kecepatan yang hampir mirip degan Naive Bayes, namun performanya masih harus bersaing lebih baik lagi dengan Naive Bayes, juga dengan mempertimbangkan kekurangan KNN untuk studi kasus dengan karakteristik data yang kompleks.


## Kesimpulan
Proyek ini mendemonstrasikan proses tahapan prediksi kepribadian berdasarkan data kuantitatif individu dari kebiasaannya yang berhubungan dengan kegiatan bersosial. Dari eksperimen ini dapat diketahui bahwa proses prediksi kepribadian seseorang dapat dilakukan dengan melatih model machine learning yang optimal, tanpa harus melakukan survey kualitatif dan membutuhkan banyak waktu tenaga ahli dalam menarik kesimpulan kepribadian.

Di antara KNN, SVM, dan Naive Bayes, model terbaik didapatkan dari melatih data dengan algoritma SVM. Namun, model dengan Naive Bayes adalah yang tercepat ketika dilihat dari segi waktu pelatihan model, tanpa mengorbankan performa model yang terlalu signifikan di bawah dua model yang lain. Oleh karena itu, proyek ini memperlihatkan bahwa algoritma terbaik untuk studi kasus ini adalah Naive Bayes, dengan performa model dan kecepatan yang sangat tinggi.


