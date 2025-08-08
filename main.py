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
    df_predict = pd.read_csv('dataset_produk.csv')
    print("Dataset 'dataset_produk.csv' berhasil dimuat.")

    # 3. Menyesuaikan nama kolom agar sesuai dengan data pelatihan
    df_predict.rename(columns={
        'protein': 'proteins',
        'total_fat': 'fat',
        'total_carbohydrate': 'carbohydrate'
    }, inplace=True)
    print("Nama kolom berhasil disesuaikan.")

    # 4. Pra-pemrosesan data (sama seperti saat pelatihan)
    def clean_numeric_data(col):
        if isinstance(col.iloc[0], str):
            return pd.to_numeric(col.str.replace('[^0-9.]', '', regex=True), errors='coerce')
        return col

    features = ['calories', 'proteins', 'fat', 'carbohydrate']

    for col in features:
        df_predict[col] = clean_numeric_data(df_predict[col])
    
    df_predict = df_predict.fillna(0)

    # 5. Menskalakan data menggunakan scaler yang sudah dilatih (TRANSFORM, bukan fit)
    X_scaled = loaded_scaler.transform(df_predict[features])

    # 6. Melakukan prediksi menggunakan model
    predictions_numeric = loaded_model.predict(X_scaled)
    
    # 7. Mengubah hasil prediksi numerik kembali menjadi grade (A, B, C, D)
    predictions_grade = pd.Series(predictions_numeric).map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})
    df_predict['nutrition_level'] = predictions_grade

    # 8. Menambahkan kolom keterangan dengan alasan dan deskripsi
    def get_full_description(grade):
        base_reason = "Grade ini ditentukan berdasarkan evaluasi kombinasi nutrisi (kalori, protein, lemak, karbohidrat) yang telah dipelajari oleh model. "
        
        if grade == 'A':
            return base_reason + 'Profil nutrisi sangat baik. Cocok sebagai bagian dari pola makan seimbang.'
        elif grade == 'B':
            return base_reason + 'Profil nutrisi baik. Mengandung nutrisi bermanfaat yang penting untuk tubuh.'
        elif grade == 'C':
            return base_reason + 'Profil nutrisi moderat. Kandungan beberapa nutrisi perlu diperhatikan dalam porsi konsumsi.'
        elif grade == 'D':
            return base_reason + 'Profil nutrisi perlu diperhatikan. Cenderung memiliki kandungan tinggi pada beberapa zat tertentu.'
        return ''

    df_predict['keterangan'] = df_predict['nutrition_level'].apply(get_full_description)

    # 9. Tampilkan hasilnya
    print("\nHasil Prediksi:")
    print(df_predict[['name', 'nutrition_level', 'keterangan']])
    df_predict.to_csv('hasil_prediksi_svm.csv', index=False)

if __name__ == '__main__':
    predict_from_csv()