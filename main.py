import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

def predict_from_csv():
    # 1. Memuat model SVM dan scaler yang sudah terlatih
    try:
        loaded_model = joblib.load('svm_model_linear.joblib')
        loaded_scaler = joblib.load('scaler.joblib')
        print("Model dan Scaler berhasil dimuat.")
    except FileNotFoundError:
        print("Error: Pastikan file '.joblib' ada di folder yang sama.")
        return

    # 2. Muat dataset yang akan diprediksi
    df_predict = pd.read_csv('nutrition.csv')
    print("Dataset 'nutrition.csv' berhasil dimuat.")

    # 3. Pra-pemrosesan data (sama seperti saat pelatihan)
    def clean_numeric_data(col):
        if isinstance(col.iloc[0], str):
            return pd.to_numeric(col.str.replace('[^0-9.]', '', regex=True), errors='coerce')
        return col

    features = ['calories', 'proteins', 'fat', 'carbohydrate']

    for col in features:
        df_predict[col] = clean_numeric_data(df_predict[col])
    
    df_predict = df_predict.fillna(0)

    # 4. Menskalakan data menggunakan scaler yang sudah dilatih (TRANSFORM, bukan fit)
    X_scaled = loaded_scaler.transform(df_predict[features])

    # 5. Melakukan prediksi menggunakan model
    predictions_numeric = loaded_model.predict(X_scaled)
    
    # 6. Mengubah hasil prediksi numerik kembali menjadi grade (A, B, C, D)
    # PENTING: Urutan grade ini harus sama dengan saat training!
    predictions_grade = pd.Series(predictions_numeric).map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})
    df_predict['nutrition_level'] = predictions_grade

    # 7. Tampilkan hasilnya
    print("\nHasil Prediksi:")
    print(df_predict[['name', 'nutrition_level']])
    df_predict.to_csv('nutrition_with_predictions.csv', index=False)

if __name__ == '__main__':
    predict_from_csv()