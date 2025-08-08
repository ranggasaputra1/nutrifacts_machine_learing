# Nutrifacts ML

Nutrifacts Machine Learning

![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg) ![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)

Repositori ini berisi kode dan dokumentasi untuk model _machine learning_ yang memprediksi grade makanan berdasarkan informasi nutrisi. Model ini dirancang untuk membantu individu membuat pilihan yang lebih tepat tentang asupan makanan mereka.

## Ringkasan Proyek

Proyek ini menggunakan **Support Vector Machine (SVM) dengan kernel Radial Basis Function (RBF)** untuk memprediksi grade makanan. Model ini dioptimalkan menggunakan **GridSearchCV** untuk menemukan _hyperparameter_ terbaik, sehingga mencapai performa maksimal.

Model ini dilatih menggunakan fitur-fitur nutrisi berikut:

- calories
- proteins
- fat
- carbohydrate

Model ini mencapai akurasi **94.06%** pada data uji, menjadikannya sangat andal untuk tugas klasifikasi.

## Instalasi

1. Klon Repositori

```sh
git clone https://github.com/dzakyadlh/nutrifacts-ml.git
```

2. Instal Pustaka

```sh
pip install -r requirements.txt
```

**Pustaka yang dibutuhkan:**

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- joblib

## Contoh Penggunaan Model untuk Prediksi

```python
import numpy as np
from joblib import load

# Memuat model dan scaler yang sudah terlatih
model = load('svm_model_linear.joblib')
scaler = load('scaler.joblib')

# Siapkan data input (urutan dan nilai harus sesuai)
# Contoh data: 100 kalori, 5g protein, 8g lemak, 25g karbohidrat
input_data = np.array([[100, 5, 8, 25]])

# Lakukan penskalaan pada data input, sama seperti saat model dilatih
input_data_scaled = scaler.transform(input_data)

# Lakukan prediksi
prediction = model.predict(input_data_scaled)

print(prediction)
```

**Urutan fitur pada input data:**

1. calories
2. proteins
3. fat
4. carbohydrate

## Pustaka yang Digunakan

| Pustaka      | Fungsi                                                                                                                                          |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Numpy        | Menyediakan komputasi numerik yang efisien dengan array dan matriks multi-dimensi.                                                              |
| Pandas       | Menawarkan struktur data berkinerja tinggi dan alat untuk analisis serta manipulasi data.                                                       |
| Seaborn      | Sebuah pustaka visualisasi Python tingkat tinggi yang dibangun di atas Matplotlib, khusus untuk plot statistik.                                 |
| Matplotlib   | Pustaka plotting Python yang komprehensif untuk membuat visualisasi statis, animasi, dan interaktif.                                            |
| Scikit-learn | Pustaka machine learning yang kuat dengan berbagai algoritma untuk supervised dan unsupervised learning.                                        |
| Joblib       | Pustaka Python untuk serialisasi dan persistensi objek Python yang efisien, sering digunakan untuk menyimpan dan memuat model machine learning. |
