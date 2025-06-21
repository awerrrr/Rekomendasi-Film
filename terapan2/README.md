# Laporan Akhir Machine Learning :  Skincare Recommendation System - Kevin Caesar
## Proyek Overview
![skincare_animasi](https://github.com/user-attachments/assets/ebefa55c-ab1d-4483-8307-7221c73a6b7c)

Skincare sudah menjadi kebutuhan pokok sama halnya dengan sandang , pangan, dan papan [[1]](https://jurnal.umsu.ac.id/index.php/MANEGGIO/article/view/17785) . Skincare merupakan produk perawatan kulit yang berfungsi untuk membantu menjaga kulit supaya tetap sehat dan terawat serta melindungi   dari   radikal   bebas   yang   akan   menyebabkan   rusaknya   lapisan epidermis  kulit [[2]](https://www.jurnal.politeknik-kebumen.ac.id/E-Bis/article/view/668/305). Keanekaragaman  produk  dan  manfaatnya  yang  beragam  terus  dikembangkan oleh pelaku usaha skincare dari segi kualitas produk dan harga untuk menarik lebih banyak konsumen [[3]](https://journal.stieamkop.ac.id/index.php/yume/article/view/1545/992). Adapun faktor lain yang harus diperhatikan karena dapat mempengaruhi keputusan konsumen, yaitu ulasan konsumen online dan penilaian konsumen online, keduanya dapat meningkatkan kepercayaan konsumen [[3]](https://journal.stieamkop.ac.id/index.php/yume/article/view/1545/992). Salah satu masalah dalam membeli produk perawatan kulit secara online adalah pengguna tidak dapat mencoba produk tersebut dan bergantung pada ulasan penilaian pelanggan lain. Namun, ulasan penilaian pada skala 1 hingga 5 dianggap tidak cukup untuk mewakili kualitas produk, dan pengguna perlu membaca teks ulasan yang ditulis oleh pengguna lain untuk mendapatkan informasi yang lebih spesifik tentang kualitas produk [[4]](https://ieeexplore.ieee.org/abstract/document/10037471). customer  atau pengguna yang  masih  awam  dalam  dunia skincaremasih mengalami  kesulitan dalami mengkatagorikan  produk,  misalnya  mereka  kurang  mengetahui mana toner dan serum. Oleh karena itu diperlukan sebuah sistem yang dapat merekomendasikan produk skincare [[5]](https://ejournal.umm.ac.id/index.php/repositor/article/view/32284/14105).

Sistem rekomendasi telah menjadi bagian yang tak terpisahkan dari hampir semua sistem berbasis informasi serta e-commerce pada umumnya digunakan agar tepat pemberian saran pelanggan kepada pelanggan. Algoritma dan kecerdasan buatan atau Artificial Intelligence (AI) kini digunakan untuk membuat rekomendasi skincare personal, mendeteksi kebutuhan skincare, atau bahkan menghasilkan skincare. Oleh karena itu, penulis ingin mengembangkan sistem rekomendasi musik dengan 2 approach, yaitu content-based approach dan collaborative approach untuk memberikan sistem rekomendasi terbaik untuk pengguna. Hal ini penting agar pengguna dapat mendapatkan skincare rekomendasi terbaik untuk mereka serta juga penting bagi brand agar mereka mampu terus berkembang untuk memberikan produk terbaik bagi konsumennya.


## Business Understanding
### Problem Statements
1. bagaimana pesebaran penggunaan skincare berdssarkan gender dan produk yang bebas cruelty free?
2. bagaimana performa brand dan product berdasatkan rating?
3. bagaimana efektivitas bahan dan performa produk menangani berbagai macam tipe kulit berdasarkan rating?
4. Bagaimana cara membuat sistem rekomendasi terbaik yang dapat diimplementasikan?

### Goals
1. Mengetahui pesebaran penggunaan skincare berdssarkan gender dan produk yang bebas cruelty free
2. Mengetahui performa brand dan product berdasatkan rating dengan menggunakan teknik visualisasi.
3. membuat visualisasi heatmap dan grafik untuk efektivitas bahan dan performa produk menangani berbagai macam tipe kulit berdasarkan rating.
4. Menggunakan algoritma cosine similarity maupun pemodelan machine learning untuk membuat sistem rekomendasi, lalu mengevaluasi menggunakan untuk menjamin keakuratan sistem rekomendasi.

### Solution Approach
Untuk mencapai goals di atas, berikut adalah pendekatan algoritma yang akan digunakan dalam pengembangan sistem rekomendasi skincare:
1. Mengimplementasikan Exploratory Data Analysis (EDA) untuk analisis dan visualisasi data.
2. Mengimplementasikan content-based filtering approach menggunakan algoritma cosine similarity.
  pendekatan ini akan merekomendasikan skincare kepada pengguna berdasarkan kesamaan fitur skin type, categori skincare, dan ingredients. Cosine Similarity akan digunakan untuk mengukur seberapa mirip satu skincare dengan skincare lain berdasarkan representasi vektornya.
3. Mengimplementasikan collaborative-based filtering approach menggunakan algoritma deep learning.
  pendekatan ini akan memanfaatkan data rating dari pengguna lain untuk memberikan rekomendasi skincare. Dengan menggunakan Deep Learning, sistem akan mencari pengguna dengan pola rating yang mirip dan merekomendasikan skincare yang disukai oleh pengguna-pengguna tersebut.
4. Evaluasi Performa Model 
  setelah model dibangun, evaluasi performa akan dilakukan menggunakan metrik seperti Precision dan Root Mean Squared Error. Ini akan memberikan wawasan tentang efektivitas model dalam merekomendasikan skincare yang relevan kepada pengguna.

## Data Understanding
Dataset yang digunakan untuk membuat sistem rekomendasi skincare pada responden diambil dari platform kaggle yang dipublikasikan oleh Waqar Ali dengan usability score 10/10. Dataset dapat diakses pada link [berikut](https://www.kaggle.com/datasets/waqi786/most-used-beauty-cosmetics-products-in-the-world/data).

### Keterangan Variabel
Dataset ini memiliki 14 variabel dengan keterangan sebagai berikut.

Variabel | Keterangan
----------|----------
user_id | nomor unik pembeli
product_name | nama produk skincare
brand | nama brand dari produk skincare
category | Jenis produk skincare (serum, highlighter, mascara, faceoil,facemask)
usage_frequency | frekuensi penggunaan produk (daily, weekly, monthly, occasional)
Rating | Nilai produk dari skala 1.0-5.0 
price_USD | Harga produk skincare
Number of reviews | jumlah ulasan produk skincare
product size | isi dari skincare dari 50mL sampai 250mL
Gender Target | Target dari produk skincare
skin type | jenis kulit (oily, normal, dry, combination, senstive)
packaging type | bentuk packaging skincare (tube, compact, spray, stick, bottle, jar)
main ingredients| bahan kandungan skincare seperti vitamin c, retinol
cruelty free | skincare yang diujicobakan pada hewan
country of origin | asal product skincare

<br>

| No | Column             | Non-Null Count | Dtype    |
|----|--------------------|----------------|----------|
| 0  | Product_Name       |15000 non-null  | object   |
| 1  | Brand              |15000 non-null  | object   | 
| 2  | Category           |15000 non-null  | object   | 
| 3  | Usage_Frequency    |15000 non-null  | object   | 
| 4  | Price_USD          |15000 non-null  | float64  |
| 5  | Rating             |15000 non-null  | float64  |
| 6  | Number_of_Reviews  |15000 non-null  | int64    |
| 7  | Product_Size       |15000 non-null  | object   | 
| 8  | Skin_Type          |15000 non-null  | object   | 
| 9  | Gender_Target      |15000 non-null  | object   | 
| 10 | Packaging_Type     |15000 non-null  | object   | 
| 11 | Main_Ingredient    |15000 non-null  | object   | 
| 12 | Cruelty_Free       |15000 non-null  | bool     | 
| 13 | Country_of_Origin  |15000 non-null  | object   | 
| 14 | user_id            |15000 non-null  | object   | 

Dari tabel di atas didapat informasi tidak terdapat nilai null pada tiap kolom untuk dataset yang digunakan:
   - Terdapat 11 fitur dengan tipe categorical atau object yaitu `Product name, brand, category, usage frequency, product size, skin type, gender target, packaging type, main ingredient, country of origin, user_id`.
   - Terdapat 3 feature dengan tipe numeric terdiri dari 2 float64 yaitu `price_usd` dan `rating` ,sedangkan 1 int64 yaitu `number_of_reviews`.
   - Terdapat 1 feature dengan tipe boolen yaitu `cruelty_free`.

### Statistik Data
Selanjutnya akan ditampilkan statistik data numerikal secara umum:
| Statistic |	Price_USD   | Rating    |Number_of_Reviews|
|-----------|-------------|-----------|-----------------|
|   count   | 15000.00000 |	15000.000	| 15000.000000 	  |
|   mean	  | 80.134108	  | 3.002327	| 5014.231333 	  |
|   std	    | 40.402983	  | 1.168029	| 2855.665464 	  |
|   min	    | 10.000000	  | 1.000000	| 52.000000   	  |
|   25%	    | 45.480000	  | 2.000000	| 2562.000000     |
|   50%	    | 80.040000	  | 3.000000	| 5002.000000     |
|   75%	    | 114.760000  | 4.000000	| 7497.000000     |
|   max	    | 149.990000  | 5.000000	| 10000.000000    |

Table diatas memberikan informasi statistik pada masing-masing kolom, antara lain:

- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval - dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.

dari tabel diatas disimpulkan bahwa:
- pada kolom `price_usd` menunjukkan bahwa harga produk skincare dengan harga paling murah 10 dollar dan paling mahal 149.9 dollar dengan rata2 keseluruhan harga produk skincare 80 dollar.
- pada kolom `rating` menunjukkan bahwa pengguna menggunakan produk memberikan rating kurang baik dengan nilai 1.0 dan memberikan sangat baik 5.0 serta rata-rata produk skincare diberikan nilai 3.0.
- pada kolom `number_of_reviews` menunjukkan bahwa review produk skincare pada suatu produk paling sedikit sebanyak 52 review dan paling banyak pada suatu produk 10000 review serta rata-rata produk skincare direview pengguna produk skincare sebanyak 5014 reviews.

### Data Cleaning
#### memeriksa data duplikasi
<img width="613" alt="check_data_duplikat" src="https://github.com/user-attachments/assets/301bee93-b51e-4bc1-abc7-32a1701b35d9">

dari hasil diatas disimpulkan bahwa data tidak terjadi duplikasi maka akan dilanjutkan pengecekan missing value pada data.

#### memeriksa missing value
<img width="203" alt="check_missing_value" src="https://github.com/user-attachments/assets/5e398d76-abf4-423f-b800-d9d0d1744397">

dari hasil diatas disimpulkan bahwa data tidak terdapat missing value pada tiap kolom maka dapat dilakukan tahan Explorasi Data Analysis dan Pemodelan


## Exploratory Data Analysis (EDA)
### Gender Target
![target_gender](https://github.com/user-attachments/assets/133f93dd-70c0-4911-b329-17e3c67eddbe)
<br>
berdasarkan diagram diatas dijelaskan bahwa perusahaan skincare membuat produk untuk segmen pertama pada konsumen pria dengan angka 33.4%, segmen kedua pada produk untuk segmen kedua pada perempuan dengan angka 33.3%, dan segmen ketiga pada unisex 33.2%. Hal ini karena pria sudah mulai sadar pentingnya penggunaan skincare dan memiliki peluang yang baik dan masi sedikit produk skincare untuk pria dibandingkan skincare perempuan.

### Brand Cruelty Free
![cruelty_free](https://github.com/user-attachments/assets/bbb6744f-ca43-40cc-a6fd-793cfd9a4599)
<br>
berdasarkan gambar diatas diketahui bahwa masih banyak produk skincare yang mengujicoba produk skincare ke sebesar sebesar 50.6%, sedangkan produk skincare yang sudah beralih dari penggunaan media hewan untuk pengujicoba produk dengan kultur sel manusia atau jaringan manusia.

### Effectiveness of Ingredients on Skin type based on Rating Score
![efektif_ingredient](https://github.com/user-attachments/assets/20d90f62-a2ea-4b15-a8a5-0fd7b4b8698e)
<br>
berasarkan heatmap diatas disimpulkan bahwa:
1. kulit bertipe `kering` menggunakan skincare umumnya mengandung bahan hylaronic acid.
2. kulit bertipe `normal` menggunakan skincare umumnya mengandung bahan retinol dan salicylic acid.
3. kulit bertipe `kombinasi` menggunakan skincare umumnya mengandung bahan glycerin dan shea butter.
4. kulit bertipe `oily` menggunakan skincare umumnya mengandung shea butter.
5. kulit bertipe `sensitive` menggunakan skincare umumnya jarang mengandung retinol.

### Avarage Rating and Number of Reviews by Skin Type
![avg_rating_and_number_by_skin_type](https://github.com/user-attachments/assets/18f5a9a9-6042-439e-a796-87a640839d01)
<br>
penjelasan:

berdasarkan grafik diatas bahwa pengguna tipe kulit normal menggunakan produk skincare rata-rata memberikan rating baik diatas 3.5 keatas ,sedangkan untuk kulit tipe oily dan tipe sensitif rata-rata memberikan rating dibawah 3.0. Grafik menunjukkan bahwa kulit tipe combination, oily, dan sensitive memiliki kepedulian terhadap kesehatan kulit dilihat dari rata-rata jumlah reviews. Dengan demikian, perusahaan skincare harus saat mengkampayekan dan edukasi produknya mengenai bahan kandungan yang harus sesuai dengan segmen sesuai tipe kulitnya karena kulit tipe combinatian,tipe kulit oily, dan kulit sensitive memiliki potensial menjadi pelanggan yang loyal dilihat dari grafik jumlah reviews jika diberikan produk skincare yang tepat karena dari rating tipe kulit oily dan tipe sensitive rata-rata memberikan rating rendah yang bermakna produk yang dibelinya kurang tepat.

### Product by Country
![country](https://github.com/user-attachments/assets/70d8131e-2208-4108-8dde-9e7a4a8750ac)
<br>
beradasarkan plot diatas dapat disimpulkan bahwa negara yang memproduksi skincare terbanyak pada negara italia, USA, dan Australia yang bermakna banyak brand produk skincare memiliki banyak jenis produk skincare karena memiliki empat musim.


### Brand and Product Based on Rating
![brand_product](https://github.com/user-attachments/assets/a1356261-2e4b-4e82-8386-52d1093a8e24)
<br>
berdasarkan plot diata dijelaskan bahwa brand memiliki reputasi yang baik ditandai dengan rating diatas rata-rata:
1. Milk Makeup yang terkenal terbuat dari bahan 100% vegan, 
2. Laura Mercier.
3. Hourglass yang merupakan 100% vergan.

tiga produk yang memiliki memiliki reputasi yang baik:
1. Perfect Powder
2. Divine Cleanser
3. Magic Powder

disimpulkan bahwa brand dengan reputasi baik oleh produk skincare 100% vegan ,sedangkan produk skincare yang paling disukai oleh responden yaitu powder.

## Data Preparation
Karena berbeda antara content-based filtering dengan collaborative filtering, maka data preparation dari kedua approach tersebut akan dilakukan secara masing-masing. Teknik Data preparation yang dilakukan terdiri dari:
- TF-IDF Vectorizer 
- Encoding Data User Rating
- Train-test-split Data User Rating

### 1. Content-Based Filtering
Untuk content-based filtering, kita akan fokus pada product_name, brand beserta category,skin,ingredient yang sudah disatukan untuk menjadi dasar pembuatan sistem rekomendasi tersebut. Oleh karena itu, dataframe hanya terdiri 4 kolom dari data yang dimiliki:
* `Product_Name`
* `Brand`
* `Brand_Product`
* `category_ingredient_skin`

Selanjutnya, digunakan TfidfVectorizer() pada kolom kombinasi category, skin type, ingredients untuk menghasilkan output berupa angka antara 0 - 1. Lalu, dibentuk dataframe yang berisi kolom kombinasi category, skin type, ingredients yang telah dilakukan vektorisasi dengan TfidfVectorizer() sebagai kolom dan seluruh nama product brand skincare sebagai barisnya. Hal ini dilakukan karena akan digunakan cosine similarity pada content-based filtering, dimana cosine similarity memerlukan bentuk angka agar dapat dihitung. Contoh dari dataframe dapat dilihat pada tabel berikut.

| brand_product | acid | aloe | bb | blush | bronzer | butter | cc | cleanser | combination | concealer | ... | retinol | salicylic | sensitive | serum | setting | shadow | shea | spray | vera | vitamin |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Drunk Elephant-Ultra Face Mask (Blush)	| 0.000000	| 0.000000	| 0.0	| 0.728084	| 0.0	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| ...	| 0.511075	| 0.000000	| 0.456832	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| 0.000000	| 0.0 |
|Laura Mercier-Ultra Lipstick (Makeup Remover) |	0.000000	|0.000000|	0.0	|0.000000	| 0.0 |	0.383847	| 0.0 |	0.0	| 0.000000 |	0.0	| ... |	0.000000 |	0.000000	| 0.000000	| 0.000000	| 0.0	| 0.0	| 0.383847	| 0.0	| 0.000000	| 0.0 | 
| Natasha Denona-Ultra Serum (Highlighter)	| 0.000000	| 0.460628	| 0.0	| 0.000000	| 0.0	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| ...	| 0.000000	| 0.000000	| 0.405964	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| 0.460628	| 0.0 |
| Ilia Beauty-Divine Serum (Face Mask)	| 0.000000	| 0.000000	| 0.0	| 0.000000	| 0.0	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| ...	| 0.000000	| 0.000000	| 0.000000	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| 0.000000	| 0.0 | 
| Charlotte Tilbury-Super Foundation (Highlighter)	| 0.000000	| 0.000000	| 0.0	| 0.000000	| 0.0	| 0.000000	| 0.0 | 	0.0	| 0.000000	| 0.0	| ...	| 0.000000	| 0.000000	| 0.000000	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| 0.000000	| 0.0 | 
| ...	| ...	| ...	| ...	| ...| 	...	| ...	| ...	| ...	| ...	| ...	| ...	| ...	| ...	| ...	| ...	| ...	| ...	| ...	| ...	| ...	| ... | 
| Patrick Ta-Magic Eyeliner (Face Mask)	| 0.000000	| 0.406752	| 0.0	| 0.000000	| 0.0	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| ...	| 0.000000	| 0.000000	| 0.358482	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| 0.406752	| 0.0 | 
| Farsali-Perfect Powder (Serum)	| 0.371885	| 0.000000	| 0.0	| 0.000000	| 0.0	| 0.000000	| 0.0	| 0.0	| 0.000000	| 0.0	| ...	| 0.000000	| 0.483455	| 0.000000	| 0.665253	| 0.0	| 0.0	| 0.000000	| 0.0	| 0.000000	| 0.0| 

**Mengapa diperlukan Mengubah data kedalam representasi numerik?**

- Data perlu diubah kedalam representasi numerik karena sistem rekomendasi berbasis konten membutuhkan representasi numerik dari teks atau fitur kategori agar dapat mengukur kemiripan antar-item. Misalnya, dalam rekomendasi skincare, kategori seperti "Oily," "Vitamin C," atau "Serum" diubah menjadi nilai numerik untuk dihitung kemiripannya.

### 2. Collaborative Filtering
Untuk collaborative filtering, kita juga akan fokus pada user_id, rating beserta category_ingredient_skin dari product skincare tersebut. Berbeda dengan content-based, kita hanya akan mengambil 3 kolom dari data yang dimiliki, yaitu
* `user_id`
* `track_name`
* `Rating`

Karena `user_id` dan `track_name` memiliki tipe data string dan unik, maka dilakukan encoding terhadap kedua kolom tersebut, kemudian dibentuk dataframe yang berisi kolom `user_id` yang sudah diencoding, kolom `track_name` yang sudah diencoding, dan `Rating`. Contoh dari dataframe dapat dilihat pada tabel berikut.

| No | track | name | Rating |
|-----|-----|-----|-----|
| 11499	| 11499	| 6721	| 4.1 | 
| 6475	| 6475	| 6282	| 1.8 | 
| 13167	| 13167	| 12418	| 3.1 | 
| 862	| 862	| 859	| 4.8 | 

**Mengapa diperlukan melakukan Encoding data?**
- Encoding data perlu dilakukan karena pada Collaborative Filtering, model harus belajar dari pola interaksi pengguna terhadap item. Data perlu diubah ke dalam bentuk numerik agar model neural network dapat memprosesnya.


Setelah data yang diperlukan telah diencoding, selanjutnya data dibagi menjadi dua, yaitu data training dan data testing untuk pembuatan dan pelatihan model. Data training digunakan untuk melatih model dengan data yang ada, sedangkan data testing digunakan untuk menguji model yang dibuat menggunakan data yang belum dilatih. Pembagian data ini dilakukan dengan perbandingan 80% : 20% untuk data training dan data testing.

**Mengapa diperlukan melakukan Train-test-split data?**

- Memisahkan data menjadi set pelatihan dan pengujian memungkinkan kita untuk mengevaluasi kinerja model pada data yang tidak pernah dilihat sebelumnya. Ini memberikan gambaran yang lebih akurat tentang seberapa baik model yg kita buat.


## Modelling and Result

### 1. Content-Based Filtering

Content-based filtering menggunakan cosine similarity sebagai algoritma untuk membuat sistem rekomendasi berdasarkan content-based filtering approach. Cosine similarity mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity. Cosine similarity dirumuskan sebagai berikut.

$$Cos (\theta) = \frac{\sum_1^n a_ib_i}{\sqrt{\sum_1^n a_i^2}\sqrt{\sum_1^n b_i^2}}$$

Pada python, kita akan menggunakan  `cosine_similarity` untuk mendapatkan nilai cosinus dua vektor dalam matriks. Cosine similarity memiliki kelebihan seperti output yang ternormalisasi (rentang -1 hingga 1) sehingga memudahkan interpretasi, penggunaan yang mudah dan sederhana, serta efisien untuk data sparse berdimensi tinggi, seperti TF-IDF. Meski demikian, cosine similarity memiliki beberapa kelemahan, seperti menganggap seluruh faktor/parameter sama penting, sensitif terhadap perubahan 'sudut vektor', dan tidak selalu cocok untuk data negatif. Setelah dibentuk sistem rekomendasi, selanjutnya akan diuji sistem rekomendasi ini untuk menampilkan top 10 rekomendasi berdasarkan  yang category, skin type, ingredient oleh user. Diperoleh hasil berikut.

`content_based_skincare_recommendations('Ilia Beauty-Perfect Powder (Concealer)')`

| No | brand_product	| Product_Name	| Brand	| category_ingredient | 
|---|---|---|---|---| 
| 0	| Urban Decay-Super Serum (Concealer) | Super Serum	| Urban Decay	| Concealer, Aloe Vera, Sensitive | 
| 1	| Danessa Myricks-Divine Makeup Remover (Concealer) | 	Divine Makeup Remover	| Danessa Myricks	| Concealer, Aloe Vera, Sensitive |
| 2	| Tarte-Divine Primer (Concealer)	| Divine Primer| 	Tarte | Concealer, Aloe Vera, Sensitive | 
| 3	| Tarte-Divine Primer (Concealer)	| Divine Primer	| Tarte	| Concealer, Vitamin C, Oily | 
| 4| 	Patrick Ta-Divine Highlighter (Concealer)| 	Divine Highlighter | 	Patrick Ta	| Concealer, Aloe Vera,Sensitive | 
| 5	| Becca-Ultra Bronzer (Concealer)	| Ultra Bronzer	| Becca	| Concealer, Aloe Vera, Sensitive | 
| 6	| E.l.f.-Divine Exfoliator (Concealer)	| Divine Exfoliator	| E.l.f.	| Concealer, Aloe Vera, Sensitive | 
| 7	| Becca-Magic Powder (Concealer) | 	Magic Powder	| Becca | Concealer, Aloe Vera, Sensitive| 
| 8	| Pat McGrath Labs-Super Highlighter (Concealer)	| Super Highlighter	| Pat McGrath Labs	| Concealer, Aloe Vera, Sensitive | 
| 9	| Ilia Beauty-Ultra Lip Gloss (Concealer) | Ultra Lip Gloss	| Ilia Beauty	| Concealer, Aloe Vera, Sensitive | 

### 2. Collaborative Filtering

Collaborative Filtering menggunakan deep learning, tepatnya embedding layer untuk membuat model deep learning. Embedding layer merupakan tipe layer pada deep learning yang digunakan untuk mentransformasikan data kategorikal menjadi vektor dengan nilai kontinu. Pada python, kita menggunakan `tensorflow.keras.layers Embedding` untuk membentuk embedding layer. Embedding Layer memiliki kelebihan seperti mengurangi kompleksitas model, dapat digunakan di berbagai macam algoritma deep learning, dan menangkap hubungan semantic pada data. Meski demikian, embedding layer juga memiliki beberapa kelemahan, seperti membutuhkan data yang banyak, sensitif terhadap hyperparameter, dan cold start problem. Setelah model dibentuk dan dilatih, diperoleh hasil `root_mean_squared_error: 0.0695` untuk data training dan `val_root_mean_squared_error:  0.2967` untuk data testing. Nilai tersebut sudah bagus untuk digunakan dalam sistem rekomendasi, sehingga dapat dibentuk sistem rekomendasi berdasarkan model tersebut. Selanjutnya, akan diuji sistem rekomendasi ini untuk menampilkan top 10 rekomendasi skincare berdasarkan  ingredient, skin type, category. Diperoleh hasil berikut.

`recommend_tracks_based_on_track_name('Morphe-Super Setting Spray (Serum)', top_n=10)`

Rekomendasi berdasarkan track dengan brand dan produk: 'Morphe-Super Setting Spray (Serum)'\
10 Rekomendasi skincare yang cocok untuk kamu:
 1. produk RMS Beauty-Divine Powder (Blush) dengan kategori Blush, Shea Butter, Combination dan rating 5.0
 2. produk Bobby Brown-Divine Face Oil (Eyeliner) dengan kategori Eyeliner, Vitamin C, Combination dan rating 4.9
 3. produk Bourjois-Ultra Face Mask (Lip Gloss) dengan kategori Lip Gloss, Shea Butter, Combination dan rating 4.9
 4. produk Danessa Myricks-Perfect Moisturizer (Powder) dengan kategori Powder, Glycerin, Dry dan rating 4.9
 5. produk Hourglass-Divine Face Mask (Face Oil) dengan kategori Face Oil, Vitamin C, Normal dan rating 5.0
 6. produk Huda Beauty-Divine Mascara (Contour) dengan kategori Contour, Glycerin, Sensitive dan rating 4.9
 7. produk Urban Decay-Perfect Cleanser (Lip Gloss) dengan kategori Lip Gloss, Shea Butter, Oily dan rating 4.8
 8. produk Perricone MD-Magic Foundation (BB Cream) dengan kategori BB Cream, Vitamin C, Combination dan rating 5.0
 9. produk Juvia’s Place-Magic Lipstick (Bronzer) dengan kategori Bronzer, Hyaluronic Acid, Dry dan rating 4.8
 10. produk Farsali-Magic Lipstick (Setting Spray) dengan kategori Setting Spray, Shea Butter, Combination dan rating 5.0

## Evaluation

### 1. Content-Based Filtering

Pada content-based filtering, model ini hanya menggunakan metrik Precision untuk mengetahui seberapa baik perforam model tersebut. Presisi adalah metrik yang biasa digunakan untuk mengevaluasi kinerja model pengelompokan. Metrik ini menghitung rasio antara nilai ground truth (nilai sebenarnya) dengan nilai prediksi yang positf. Perhitungan rasio ini dijabarkan melalui rumus di bawah ini:

$$ Precision = \frac{TP}{TP + FP} $$

Dimana:

- TP (*True Positive*), jumlah kejadian positif yang diprediksi dengan benar.
- FP (*False Positive*), jumlah kejadian positif yang diprediksi dengan salah.

Berdasarkan hasil yang terdapat pada tahap Model and Result dapat dilihat bahwasanya besar presisi jika dihitung adalah 9/10 untuk rekomendasi Top-10. Ini menunjukan sistem mampu memberikan rekomendasi sesuai dengan skintype, category, dan ingredient.


### 2. Collaborative Filtering

Pada collaborative filtering, metrik evaluasi yang digunakan adalah Root Mean Squared Error (RMSE). 

#### Sekilas tentang RMSE

Root Mean Squared Error (RMSE) merupakan salah satu metode untuk menghitung error pada pelatihan model dengan cara menghitung jarak rata-rata antara nilai yang diprediksi dengan nilai sesungguhnya. RMSE dirumuskan sebagai berikut.

$$RMSE = \sqrt{\frac{\sum_{i=1}^n{(y_i - \hat{y_i})}^2}{N}}$$

Keterangan:
* $y_i$: Nilai sesungguhnya pada observasi ke-i
* $\hat{y_i}$: Nilai prediksi pada observasi ke-i
* $N$: Jumlah observasi

Jika nilai prediksi sangat mendekati nilai sesungguhnya, maka nilai dari $(y_i - \hat{y_i})$ akan semakin mengecil. Artinya, semakin kecil nilai dari RSME atau bahkan mendekati nol, maka model yang digunakan telah akurat dan baik.

#### Penerapan Evaluasi Model dengan RMSE

Pada collaborative filtering, setelah melatih model sebanyak 60 epoch, diperoleh hasil `RMSE = 0.0695` untuk data training dan `RMSE = 0.2967` untuk data testing. Jika dilihat menggunakan grafik, diperoleh plot sebagai berikut.

![matrik_evaluasi](https://github.com/user-attachments/assets/bf2b0aa7-c71c-4a27-abb1-3493b6437eb5)

Dari gambar tersebut, terlihat bahwa nilai RMSE pada data training selalu menurun, sementara nilai RSME pada data testing awalnya menurun, namun setelah 10 epoch, nilai RSME mulai stagnan. Meski RSME pada data testing lebih besar dari data training, namun karena keduanya telah bernilai sangat mendekati 0, maka model yang digunakan telah baik dan akurat untuk membuat sistem rekomendasi.


## Kesimpulan
1. Pria sudah mulai sadar pentingnya penggunaan skincare dan memiliki peluang yang baik dan masi sedikit produk skincare untuk pria dibandingkan skincare perempuan, dibuktikan dengan memiliki nilai pesebaran tertinggi, yaitu 33.4% dan  produk skincare yang sudah beralih dari penggunaan media hewan untuk pengujicoba produk dengan kultur sel manusia atau jaringan manusia semakin meningkat.
2. Brand skincare dengan reputasi baik oleh produk skincare 100% vegan ,sedangkan produk skincare yang paling disukai oleh responden yaitu powder.
3. perusahaan skincare harus saat mengkampayekan dan edukasi produknya mengenai bahan kandungan yang harus sesuai dengan segmen sesuai tipe kulitnya karena kulit tipe combinatian tipe kulit oily, dan kulit sensitive memiliki potensial menjadi pelanggan yang loyal dilihat dari grafik jumlah reviews jika diberikan produk skincare yang tepat karena dari rating tipe kulit oily dan tipe sensitive rata-rata memberikan rating rendah yang bermakna produk yang dibelinya kurang tepat. dengan memperhatikan ingredient skincare:
    * kulit bertipe `kering` menggunakan skincare umumnya mengandung bahan hylaronic acid.
    * kulit bertipe `normal` menggunakan skincare umumnya mengandung bahan retinol dan salicylic acid.
    * kulit bertipe `kombinasi` menggunakan skincare umumnya mengandung bahan glycerin dan shea butter.
    * kulit bertipe `oily` menggunakan skincare umumnya mengandung shea butter.
    * kulit bertipe `sensitive` menggunakan skincare umumnya jarang mengandung retinol.
4. Sistem rekomendasi dapat diimplementasikan dengan menggunakan 2 approach, yaitu content-based filtering approach menggunakan cosine similarity dan collaborativer filtering approach menggunakan embedding layer untuk memberikan sistem rekomendasi terbaik.

## Referensi

[[1]](https://jurnal.umsu.ac.id/index.php/MANEGGIO/article/view/17785) F. A. Nasution, "Keputusan Pembelian: Peranan Motivasi persepsi pembelajaran Pembelian Skincare," Maneggio: Jurnal Ilmiah Magister Manajemen, vol. 6, no. 2, pp. 193–202, 2023.

 [[2]](https://www.jurnal.politeknik-kebumen.ac.id/E-Bis/article/view/668/305) Dinda Dwi Guntari and Prihartono Aksan Halim, “Pengaruh Kualitas dan Desain Produk Terhadap Keputusan Pembelian (Survey pada Produk Envygreen Skincare)”, E-Bis, vol. 5, no. 2, pp. 295-307, Oct. 2021.

[[3]](https://journal.stieamkop.ac.id/index.php/yume/article/view/1545/992) D. W. Robi'ah and M. Nopiana, "Pengaruh Persepsi Harga dan Kualitas Produk Terhadap Keputusan Pembelian Produk Skincare Avoskin," YUME: Journal of Management, vol. 5, no. 1, pp. 433–441, 2022.

[[4]](https://ieeexplore.ieee.org/abstract/document/10037471) C. Qalbyassalam, R. F. Rachmadi, and A. Kurniawan, "Skincare Recommender System Using Neural Collaborative Filtering with Implicit Rating," in Proc. 2022 Int. Conf. Computer Engineering, Network, and Intelligent Multimedia (CENIM), Surabaya, Indonesia, 2022, pp. 272–277, doi: 10.1109/CENIM56801.2022.10037471.

[[5]](https://ejournal.umm.ac.id/index.php/repositor/article/view/32284/14105) E. Ayuningrum, Y. Azhar, and G. I. Marthasari, “Sistem Rekomendasi Produk Skincare Korea Berbasis Web Menggunakan Metode Collaborative Filtering”, JR, vol. 4, no. 4, Feb. 2024.

[6] Dicoding. Diakses pada 30 Oktober 2024 dari https://www.dicoding.com/academies/319-machine-learning-terapan

[7] Kaggle. Diakses pada 01 Desember 2024 dari https://www.kaggle.com/code/slaymaneslime/marketing-approach-beauty-comestics-analysis
