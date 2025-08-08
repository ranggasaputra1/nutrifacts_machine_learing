import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load dataset
nutrition_df = pd.read_csv("nutrition.csv")

# Ambil fitur yang diperlukan
X = nutrition_df[['calories', 'proteins', 'fat', 'carbohydrate']].copy()
product_names = nutrition_df['name']

# Normalisasi sesuai metode dalam model Anda
scaler_bad = MinMaxScaler(feature_range=(1, 10))
scaler_good = MinMaxScaler(feature_range=(1, 5))

X['calories'] = scaler_bad.fit_transform(X[['calories']])
X['fat'] = scaler_bad.fit_transform(X[['fat']])
X['carbohydrate'] = scaler_bad.fit_transform(X[['carbohydrate']])
X['proteins'] = scaler_good.fit_transform(X[['proteins']])

# Load model SVM
model = joblib.load("svm_model_linear2.joblib")
predicted = model.predict(X)

# Konversi label prediksi
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
predicted_labels = [label_map.get(i, '-') for i in predicted]

# Gabungkan hasil prediksi ke data asli
results_df = nutrition_df[['name', 'calories', 'proteins', 'fat', 'carbohydrate']].copy()
results_df['prediksi_SVM'] = predicted_labels

# Simpan 10 data awal untuk keperluan validasi manual
results_df.head(10).to_csv("hasil_prediksi_sample.csv", index=False)
print("Berhasil menyimpan hasil prediksi ke hasil_prediksi_sample.csv")
