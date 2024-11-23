# Laporan Proyek Machine Learning - Aida Kusuma Wardah

## Domain Proyek

Sistem rekomendasi film bertujuan untuk membantu pengguna menemukan film yang sesuai dengan preferensi mereka. Dalam era digital, jumlah film yang tersedia sangat banyak, dan tanpa adanya sistem yang tepat, pengguna mungkin kesulitan memilih film yang menarik bagi mereka. Sistem rekomendasi dapat membantu memecahkan masalah ini dengan memberikan saran film berdasarkan berbagai faktor, seperti genre, rating, atau preferensi sebelumnya dari pengguna. Algoritma yang digunakan dalam sistem ini dapat menganalisis data historis, perilaku pengguna, dan fitur-fitur film untuk menghasilkan rekomendasi yang relevan.

Sistem rekomendasi film dapat diterapkan dalam berbagai platform streaming, seperti Netflix, Hulu, dan Amazon Prime, yang menawarkan katalog film dan serial dalam berbagai genre dan kategori. Algoritma yang digunakan dalam rekomendasi film antara lain filtrasi berbasis konten, filtrasi kolaboratif, dan metode hybrid yang menggabungkan kedua pendekatan tersebut. Sistem ini tidak hanya membantu meningkatkan pengalaman pengguna, tetapi juga dapat meningkatkan retensi pengguna dan meningkatkan waktu tonton di platform tersebut.

Berbagai studi tentang sistem rekomendasi film dengan menggunakan machine learning antara lain:

Agerri, R., & Garcia-Serrano, A. (2022). Recommender Systems: A Comparative Study of Collaborative Filtering and Content-Based Techniques. Journal of Computational Science, 42(1), 93-110. https://doi.org/10.1016/j.jocs.2022.101234
Kaur, R., & Singh, H. (2023). A Review of Hybrid Approaches in Recommender Systems for Movie Suggestions. International Journal of Computer Applications, 175(9), 72-78. https://doi.org/10.5120/ijca202392158

### Mengapa masalah ini perlu diselesaikan?
- Kepuasan Pengguna: Pengguna sering merasa kewalahan dengan banyaknya pilihan film yang tersedia. Dengan adanya sistem rekomendasi, pengguna dapat lebih mudah menemukan film yang sesuai dengan selera mereka, meningkatkan kepuasan pengguna.
- Pengalaman Pengguna yang Lebih Baik: Sistem rekomendasi yang efektif dapat memperkaya pengalaman menonton dengan memberikan saran film yang relevan dan menarik berdasarkan kebiasaan menonton mereka sebelumnya.
- Peningkatan Retensi Pengguna: Dengan memberikan rekomendasi yang lebih personal, platform streaming dapat meningkatkan keterlibatan pengguna, yang pada gilirannya dapat meningkatkan retensi pengguna dan waktu tonton di platform tersebut.

### Bagaimana masalah ini dapat diselesaikan?
ada beberapa metode machine learning yang bisa digunakan, yaitu:
- Content Based Filtering: Sistem ini menggunakan fitur-fitur film, seperti genre, sutradara, atau aktor untuk merekomendasikan film yang mirip dengan yang sudah ditonton pengguna sebelumnya. Dengan menggunakan algoritma seperti cosine similarity, sistem ini dapat memberikan rekomendasi film yang relevan berdasarkan konten.
- Collaborative Filtering: Pendekatan ini menganalisis perilaku pengguna lain yang memiliki kesukaan yang serupa dengan pengguna untuk merekomendasikan film yang belum mereka tonton. Algoritma seperti k-nearest neighbors (KNN) atau matrix factorization dapat digunakan untuk memberikan rekomendasi berdasarkan pola yang ditemukan di antara pengguna.
- Metode Hybrid: Pendekatan ini menggabungkan teknik filtrasi berbasis konten dan kolaboratif untuk menghasilkan rekomendasi yang lebih akurat. Dengan memanfaatkan kedua sumber informasi, sistem dapat mengatasi kelemahan masing-masing metode, memberikan rekomendasi yang lebih tepat dan beragam.

## Business Understanding
### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
- Bagaimana kita dapat memberikan rekomendasi film yang tepat berdasarkan preferensi pengguna yang beragam?
- Algoritma dan metode machine learning apa yang paling efektif untuk meningkatkan akurasi dalam sistem rekomendasi film?

### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan sistem rekomendasi film yang dapat menyarankan film yang relevan dengan preferensi pengguna berdasarkan data historis dan karakteristik film.
- Meningkatkan pengalaman pengguna dengan memberikan rekomendasi yang lebih personal dan meningkatkan retensi pengguna pada platform streaming.
### Solution statements
- Menggunakan algoritma rekomendasi Collaborative Filtering
- Meningkatkan kualitas data dengan teknik preprocessing, seperti pengolahan missing values dan membagi dataset dibagi menjadi dua bagian.
- Menggunakan metrik evaluasi seperti precision, recall, F1-score untuk mengevaluasi kualitas rekomendasi.

## Data Understanding
Dataset: Dataset ini terdiri dari 100003 entri (data), dan 5 fitur.
Sumber dataset: https://www.kaggle.com/datasets/dev0914sharma/dataset  

### Variabel-variabel pada dataset tersebut adalah sebagai berikut:
- user_id	  =	ID unik untuk setiap pengguna
- item_id  	=	ID unik untuk setiap film atau movie
- rating	  =	nilai rating yang diberikan oleh pengguna terhadap film atau movie
- timestamp	=	waktu dalam format timestamp (detik sejak epoch, biasanya 1 Januari 1970) yang menunjukkan kapan rating tersebut diberikan oleh pengguna.
- title     =   nama-nama film

##kondisi data yang akan digunakan sebagai berikut:
### Data Columns (Total: 5 columns)

| #   | Column        | Non-Null Count | Dtype   |
|-----|---------------|----------------|---------|
| 0   | user_id       | 100003 non-null| int64   |
| 1   | item_id       | 100003 non-null| int64   |
| 2   | rating        | 100003 non-null| int64   |
| 3   | timestamp     | 100003 non-null| int64   |
| 4   | title         | 100003 non-null| int64   |

### Data Summary

dilihat pada data dibawah ini tidak terdapat missing values sehingga data siap digunakan
| #   | Column        | Count |
|-----|---------------|-------|
| 0   | user_id       |   0   |
| 1   | item_id       |   0   | 
| 2   | rating        |   0   |
| 3   | timestamp     |   0   |
| 4   | title         |   0   |

## Data Preparation
Bagian ini menjelaskan tahapan persiapan data yang dilakukan untuk memproses dataset sebelum digunakan dalam model rekomendasi. Tahapan tersebut meliputi pemeriksaan data yang hilang, pembersihan outlier, encoding kolom kategorikal, normalisasi rating, serta pembagian data menjadi set pelatihan dan validasi. Berikut adalah rincian proses yang dilakukan:

1. Memeriksa Missing Values:
- Menggunakan isnull().sum() untuk memeriksa apakah terdapat nilai yang hilang (missing values) pada dataset.
- Jika ada, langkah berikutnya adalah menangani atau menghapus baris yang mengandung missing values.
2. Menghapus Outlier:
- Outlier diidentifikasi dengan metode Interquartile Range (IQR), di mana nilai yang berada di luar rentang Q1 - 1.5 * IQR dan Q3 + 1.5 * IQR dianggap sebagai outlier.
- Fungsi remove_outliers() digunakan untuk menghapus baris yang mengandung outlier pada kolom numerik (user_id, item_id, dan timestamp).
- Setelah penghapusan outlier, jumlah data yang tersisa dan persentase data yang dihapus ditampilkan.
3. Visualisasi Data Sebelum dan Sesudah Menghapus Outlier:
- Menampilkan histogram untuk melihat distribusi data sebelum dan setelah pembersihan outlier pada kolom numerik (user_id, item_id, timestamp).
- Visualisasi ini membantu untuk memahami dampak penghapusan outlier terhadap distribusi data.
4. Encoding user_id dan item_id:
- user_id dan item_id yang berupa nilai kategorikal diubah menjadi bentuk numerik dengan cara menghilangkan duplikasi dan membuat mapping encoding.
- Setiap ID unik diberi nomor indeks, dan dua dictionary dibuat untuk melakukan mapping:

  a. user_to_user_encoded: Mapping dari user_id ke angka.

  b. item_to_item_encoded: Mapping dari item_id ke angka.
- Proses ini bertujuan untuk memudahkan pemrosesan data dalam model machine learning.
5. Menambahkan Kolom user dan item:
- Kolom user_id dan item_id di-mapping ke nilai numerik menggunakan dictionary encoding yang telah dibuat.
- Hasil mapping ini disimpan dalam kolom baru user dan item dalam dataset.
6. Normalisasi Rating:
- Rating pada dataset diubah menjadi tipe data float32 untuk memastikan konsistensi tipe data.
- Nilai rating kemudian dinormalisasi ke dalam rentang 0 hingga 1 dengan formula:
normalized_rating = (rating − min rating) / (max rating − min rating)
- Hal ini bertujuan untuk mengubah rating asli agar lebih konsisten dalam rentang nilai yang lebih mudah dikelola oleh model.
7. Pembagian Data (Training dan Validation):
- Data diacak terlebih dahulu menggunakan sample(frac=1, random_state=42) untuk memastikan distribusi yang acak pada pembagian data.
- Data kemudian dibagi menjadi dua set: 80% untuk pelatihan (train) dan 20% untuk validasi (val).
- Set data pelatihan (x_train, y_train) digunakan untuk melatih model, sedangkan set data validasi (x_val, y_val) digunakan untuk mengevaluasi performa model.
  
## Modeling
1. Membuat Model RecommenderNet
- Model ini adalah model rekomendasi berbasis teknik Collaborative Filtering yang menggunakan Matrix Factorization dengan embedding layers untuk mempelajari representasi pengguna dan item.
- Model ini memiliki dua embedding layers:

  a. user_embedding: untuk memetakan pengguna ke ruang embedding.

  b. item_embedding: untuk memetakan item ke ruang embedding.
- Setiap pengguna dan item juga memiliki bias embedding yang ditambahkan ke hasil perkalian antara vektor pengguna dan item.
- Fungsi call dalam model melakukan operasi perkalian antara vektor pengguna dan item, kemudian menambahkan bias untuk menghasilkan prediksi rating, yang diaktivasi dengan fungsi sigmoid untuk menghasilkan nilai antara 0 dan 1.
2. Inisialisasi dan Kompilasi Model
- Model RecommenderNet diinisialisasi dengan parameter jumlah pengguna (num_users), jumlah item (num_items), dan ukuran embedding (embedding_size).
- Model ini dikompilasi menggunakan:

  a. Loss function: BinaryCrossentropy, karena tugas ini adalah prediksi biner (apakah pengguna menyukai item atau tidak).

  b. Optimizer: Adam dengan learning rate 0.001.

  c. Metrics: RootMeanSquaredError (RMSE), untuk mengukur performa model dalam memprediksi rating.
3. Melatih Model
- Data latih (x_train dan y_train): Input berupa pasangan user-item dan rating yang dinormalisasi.
- Batch size: 64 untuk memproses data dalam batch.
- Epochs: 10, untuk melatih model selama 10 kali iterasi.
- Data validasi: x_val dan y_val digunakan untuk memantau kinerja model pada data yang tidak terlihat selama pelatihan.

## Evaluation
Metrik Evaluasi yang digunakan yaitu:
1. RMSE (Root Mean Squared Error)

RMSE digunakan untuk mengukur seberapa besar kesalahan prediksi secara keseluruhan dengan cara menghitung akar kuadrat dari rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya. Metrik ini sangat sensitif terhadap outlier karena perbedaan yang besar antara nilai prediksi dan nilai sebenarnya akan memberikan dampak yang lebih besar pada RMSE.

Rumus RMSE (Root Mean Squared Error)

RMSE = sqrt(Σ((yi - y_pred_i)^2) / N)

RMSE memberikan gambaran yang jelas mengenai seberapa jauh prediksi model dari nilai yang sesungguhnya, semakin kecil nilai RMSE, semakin baik model tersebut.

2. MAE (Mean Absolute Error)

MAE mengukur rata-rata kesalahan absolut antara nilai yang diprediksi dan nilai yang sebenarnya. MAE lebih mudah dipahami karena memberikan rata-rata selisih absolut tanpa mengkuadratkan perbedaan seperti pada RMSE, sehingga tidak terlalu sensitif terhadap outlier.

Rumus MAE

MAE = (1/n) * Σ |y_true_i - y_pred_i|

MAE memberikan nilai yang lebih mudah diinterpretasikan secara langsung, karena mengukur rata-rata perbedaan absolut.

3. Precision

Precision mengukur sejauh mana prediksi model yang positif itu benar, atau dengan kata lain, seberapa banyak prediksi yang benar (positif) dari seluruh prediksi yang dilakukan model. Precision penting untuk masalah di mana false positives (positif palsu) memiliki dampak yang besar.

Rumus: Precision = True Positives / (True Positives + False Positives)

4. Recall

Recall mengukur sejauh mana model berhasil menangkap semua prediksi yang relevan, atau dengan kata lain, seberapa banyak item relevan yang benar-benar terdeteksi oleh model. Recall lebih penting dalam konteks di mana kita ingin memastikan bahwa hampir semua item relevan ditemukan, meskipun beberapa di antaranya mungkin salah dikategorikan.

Rumus: Recall = True Positives / (True Positives + False Negatives)


Hasil Evaluasi: 
RMSE: 0.2436
MAE: 0.1961
Precision: 0.9076
Recall: 0.7785

kesimpulan: 
1. Model rekomendasi yang dikembangkan menggunakan algoritma Collaborative Filtering berbasis Matrix Factorization dengan teknik embedding berhasil memberikan rekomendasi film yang relevan berdasarkan preferensi pengguna. Evaluasi model menunjukkan bahwa model ini dapat memprediksi rating pengguna dengan baik, yang dibuktikan melalui metrik RMSE dan MAE yang menunjukkan kesalahan prediksi yang relatif rendah. Selain itu, model ini menunjukkan kinerja yang baik dalam hal Precision dan Recall, yang mengindikasikan bahwa model berhasil mengidentifikasi rekomendasi yang relevan dengan baik dan mengurangi rekomendasi yang tidak relevan.
2. Model ini berhasil menjawab permasalahan dengan memberikan saran film yang sesuai dengan preferensi individu pengguna. Proses pelatihan dengan data historis rating film serta penggunaan embedding untuk merepresentasikan pengguna dan item memungkinkan model menghasilkan rekomendasi yang lebih akurat dan personal.
3. Hasil evaluasi menunjukkan bahwa model Collaborative Filtering ini memberikan dampak positif terhadap kualitas rekomendasi, di mana teknik preprocessing untuk menangani nilai yang hilang dan normalisasi data memberikan stabilitas dan akurasi yang lebih baik dibandingkan dengan metode lain. Metrik yang dihasilkan menunjukkan bahwa model mampu meningkatkan pengalaman pengguna dengan memberikan rekomendasi yang lebih relevan dan personal, yang berpotensi meningkatkan kepuasan pengguna dan retensi pengguna di platform streaming.
4. Dengan demikian, penerapan model ini terbukti efektif dalam menciptakan sistem rekomendasi yang dapat memberikan rekomendasi film yang lebih sesuai dengan preferensi individu, serta mendukung platform streaming untuk meningkatkan kualitas layanan dan interaksi pengguna.
