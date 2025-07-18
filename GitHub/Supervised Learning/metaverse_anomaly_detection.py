#!/usr/bin/env python3
"""
Skrip Deteksi Anomali Transaksi Metaverse
==========================================

Skrip ini menggunakan machine learning untuk mendeteksi anomali dalam transaksi metaverse
berdasarkan berbagai fitur seperti amount, location, user behavior, dan risk score.

Author: Muhammad Nabil Hanif
Dataset: metaverse_transactions_dataset.csv
"""

# 1. Impor Pustaka
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTENC
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DETEKSI ANOMALI TRANSAKSI METAVERSE")
print("="*60)

# 2. Muat Data
print("\n1. MEMUAT DATA...")
print("-" * 30)

# Muat dataset dengan parsing timestamp
df = pd.read_csv('metaverse_transactions_dataset.csv', parse_dates=['timestamp'])
print(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# 3. Gambaran Umum Data Awal
print("\n2. GAMBARAN UMUM DATA AWAL")
print("-" * 30)

print("\n📊 5 Baris Pertama:")
print(df.head())

print("\n📋 Informasi DataFrame:")
print(df.info())

print("\n📈 Statistik Deskriptif untuk Fitur Numerik:")
print(df.describe())

print("\n🔢 Jumlah Nilai Unik per Kolom:")
print(df.nunique())

print("\n❓ Jumlah Nilai yang Hilang per Kolom:")
print(df.isnull().sum())

print("\n🎯 Distribusi Kelas Target (anomaly):")
print(df['anomaly'].value_counts())

# 4. Rekayasa Fitur (Feature Engineering)
print("\n3. REKAYASA FITUR")
print("-" * 30)

# Ekstrak fitur dari timestamp
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
df['month'] = df['timestamp'].dt.month

# Buat fitur is_weekend (1 jika Sabtu/Minggu, 0 jika tidak)
df['is_weekend'] = ((df['day_of_week'] == 5) | (df['day_of_week'] == 6)).astype(int)

print(f"✅ Fitur 'day_of_week' ditambahkan (0=Senin, 6=Minggu)")
print(f"✅ Fitur 'month' ditambahkan (1-12)")
print(f"✅ Fitur 'is_weekend' ditambahkan (1=weekend, 0=weekday)")

# Hapus kolom timestamp setelah ekstraksi fitur
df = df.drop('timestamp', axis=1)
print("✅ Kolom 'timestamp' dihapus setelah ekstraksi fitur")

print(f"\nUkuran dataset setelah feature engineering: {df.shape}")

# 5. Pemisahan Fitur dan Target
print("\n4. PEMISAHAN FITUR DAN TARGET")
print("-" * 30)

# Pisahkan fitur (X) dari target (y)
X = df.drop('anomaly', axis=1)
y = df['anomaly']

print(f"Jumlah fitur (X): {X.shape[1]}")
print(f"Ukuran dataset: {X.shape[0]} sampel")

# Encode target variable menggunakan LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n🏷️  Mapping Label Target:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {class_name} → {i}")

print(f"\nDistribusi target setelah encoding:")
print(pd.Series(y_encoded).value_counts().sort_index())

# 6. Pipeline Pra-pemrosesan
print("\n5. PIPELINE PRA-PEMROSESAN")
print("-" * 30)

# Definisikan fitur numerik dan kategorikal
numerical_features = ['amount', 'hour_of_day', 'login_frequency', 'session_duration', 
                     'risk_score', 'day_of_week', 'month', 'is_weekend']

categorical_features = ['transaction_type', 'location_region', 'ip_prefix', 
                       'purchase_pattern', 'age_group', 'sending_address', 'receiving_address']

print(f"📊 Fitur Numerik ({len(numerical_features)}): {numerical_features}")
print(f"📂 Fitur Kategorikal ({len(categorical_features)}): {categorical_features}")

# Buat ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Fit dan transform fitur
print("\n🔄 Melakukan pra-pemrosesan data...")
X_processed = preprocessor.fit_transform(X)
print(f"✅ Data berhasil dipra-proses. Ukuran setelah preprocessing: {X_processed.shape}")

# 7. Pembagian Data Latih-Uji
print("\n6. PEMBAGIAN DATA LATIH-UJI")
print("-" * 30)

# Split data dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"📚 Data Pelatihan: {X_train.shape[0]} sampel")
print(f"🧪 Data Pengujian: {X_test.shape[0]} sampel")
print(f"📊 Rasio Train:Test = 80:20")

# 8. Penanganan Ketidakseimbangan Kelas
print("\n7. PENANGANAN KETIDAKSEIMBANGAN KELAS")
print("-" * 30)

print("📊 Distribusi kelas SEBELUM SMOTE:")
train_distribution_before = Counter(y_train)
for label, count in sorted(train_distribution_before.items()):
    class_name = label_encoder.inverse_transform([label])[0]
    percentage = (count / len(y_train)) * 100
    print(f"  {class_name} (label {label}): {count} sampel ({percentage:.1f}%)")

# Identifikasi indeks kolom kategorikal untuk SMOTENC
# Setelah ColumnTransformer, fitur numerik ada di awal, diikuti fitur kategorikal yang di-one-hot encode
num_numerical_features = len(numerical_features)
categorical_indices = list(range(num_numerical_features, X_processed.shape[1]))

print(f"\n🔍 Indeks kolom kategorikal untuk SMOTENC: {len(categorical_indices)} kolom")

# Terapkan SMOTENC
print("\n🔄 Menerapkan SMOTENC untuk mengatasi ketidakseimbangan kelas...")
smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)

print("📊 Distribusi kelas SETELAH SMOTE:")
train_distribution_after = Counter(y_train_resampled)
for label, count in sorted(train_distribution_after.items()):
    class_name = label_encoder.inverse_transform([label])[0]
    percentage = (count / len(y_train_resampled)) * 100
    print(f"  {class_name} (label {label}): {count} sampel ({percentage:.1f}%)")

print(f"\n✅ Data pelatihan setelah SMOTE: {X_train_resampled.shape[0]} sampel")

# 9. Pelatihan Model
print("\n8. PELATIHAN MODEL")
print("-" * 30)

# Inisialisasi dan latih RandomForestClassifier
print("🌲 Melatih Random Forest Classifier...")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_resampled, y_train_resampled)
print("✅ Model berhasil dilatih!")

# 10. Evaluasi Model
print("\n9. EVALUASI MODEL")
print("-" * 30)

# Prediksi pada data test
y_pred = rf_model.predict(X_test)

print("📊 CLASSIFICATION REPORT:")
print("=" * 50)
target_names = [f"{name} ({i})" for i, name in enumerate(label_encoder.classes_)]
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n📊 CONFUSION MATRIX:")
print("=" * 30)
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n" + "="*60)
print("⚠️  CATATAN PENTING: Skor F1 untuk kelas 'high_risk' dan 'moderate_risk'")
print("    adalah metrik kinerja yang paling kritis karena ketidakseimbangan kelas.")
print("    Focus utama adalah mendeteksi transaksi berisiko tinggi dan sedang.")
print("="*60)

# 11. Fungsi Prediksi
def predict_anomaly(new_transaction_data_series, preprocessor, model, label_encoder, original_columns):
    """
    Memprediksi tingkat risiko anomali untuk transaksi baru
    
    Args:
        new_transaction_data_series (pd.Series): Data transaksi baru dengan nama kolom asli
        preprocessor: ColumnTransformer yang telah di-fit
        model: Model yang telah dilatih
        label_encoder: LabelEncoder untuk target
        original_columns: List kolom asli sebelum preprocessing
    
    Returns:
        str: Prediksi label ('low_risk', 'moderate_risk', 'high_risk')
    """
    # Konversi Series ke DataFrame 1 baris
    new_transaction_df = pd.DataFrame([new_transaction_data_series], columns=original_columns)
    
    # Pra-proses data baru
    new_transaction_processed = preprocessor.transform(new_transaction_df)
    
    # Prediksi
    prediction_encoded = model.predict(new_transaction_processed)[0]
    
    # Konversi kembali ke label asli
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    
    return prediction_label

# Simpan kolom asli untuk fungsi prediksi
original_columns = X.columns.tolist()

print("\n10. FUNGSI PREDIKSI")
print("-" * 30)
print("✅ Fungsi predict_anomaly() telah dibuat")
print("   Input: pandas Series dengan data transaksi baru")
print("   Output: Label risiko ('low_risk', 'moderate_risk', 'high_risk')")

# 12. Contoh Prediksi
print("\n11. CONTOH PREDIKSI")
print("-" * 30)

# Contoh 1: Transaksi Low Risk
print("🔍 CONTOH 1: Transaksi Berisiko Rendah")
print("-" * 40)
low_risk_example = pd.Series({
    'hour_of_day': 14,
    'sending_address': '0x1234567890abcdef',
    'receiving_address': '0xfedcba0987654321',
    'amount': 150.0,
    'transaction_type': 'transfer',
    'location_region': 'Europe',
    'ip_prefix': '192.168',
    'login_frequency': 8,
    'session_duration': 120,
    'purchase_pattern': 'focused',
    'age_group': 'established',
    'risk_score': 15.0,
    'day_of_week': 1,  # Tuesday
    'month': 6,  # June
    'is_weekend': 0
})

prediction_1 = predict_anomaly(low_risk_example, preprocessor, rf_model, label_encoder, original_columns)
print(f"📋 Input: Transfer €150, established user, risk_score=15.0")
print(f"🎯 Prediksi: {prediction_1}")

# Contoh 2: Transaksi High Risk
print("\n🔍 CONTOH 2: Transaksi Berisiko Tinggi")
print("-" * 40)
high_risk_example = pd.Series({
    'hour_of_day': 3,  # Jam tidak biasa
    'sending_address': '0xsuspicious123456',
    'receiving_address': '0xmalicious789012',
    'amount': 1500.0,  # Jumlah tinggi
    'transaction_type': 'phishing',  # Jenis transaksi mencurigakan
    'location_region': 'Asia',
    'ip_prefix': '10.0',
    'login_frequency': 1,  # Login frequency rendah
    'session_duration': 15,  # Session duration pendek
    'purchase_pattern': 'random',
    'age_group': 'new',  # User baru
    'risk_score': 95.0,  # Risk score tinggi
    'day_of_week': 6,  # Sunday
    'month': 12,  # December
    'is_weekend': 1
})

prediction_2 = predict_anomaly(high_risk_example, preprocessor, rf_model, label_encoder, original_columns)
print(f"📋 Input: Phishing €1500, new user, risk_score=95.0, jam 3 AM")
print(f"🎯 Prediksi: {prediction_2}")

# Contoh 3: Transaksi Moderate Risk
print("\n🔍 CONTOH 3: Transaksi Berisiko Sedang")
print("-" * 40)
moderate_risk_example = pd.Series({
    'hour_of_day': 22,  # Jam malam
    'sending_address': '0xmoderate567890',
    'receiving_address': '0xstandard234567',
    'amount': 750.0,  # Jumlah sedang-tinggi
    'transaction_type': 'purchase',
    'location_region': 'Asia',  # Region dengan risiko lebih tinggi
    'ip_prefix': '172.0',
    'login_frequency': 4,  # Login frequency sedang
    'session_duration': 45,  # Session duration sedang
    'purchase_pattern': 'random',  # Pattern acak
    'age_group': 'new',  # User baru
    'risk_score': 65.0,  # Risk score sedang
    'day_of_week': 4,  # Friday
    'month': 3,  # March
    'is_weekend': 0
})

prediction_3 = predict_anomaly(moderate_risk_example, preprocessor, rf_model, label_encoder, original_columns)
print(f"📋 Input: Purchase €750, new user Asia, risk_score=65.0, jam 22:00")
print(f"🎯 Prediksi: {prediction_3}")

print("\n" + "="*60)
print("🎉 SKRIP DETEKSI ANOMALI TRANSAKSI METAVERSE SELESAI!")
print("="*60)
print("\n📝 RINGKASAN:")
print(f"   • Dataset: {df.shape[0]} transaksi dengan {len(original_columns)} fitur")
print(f"   • Model: Random Forest dengan {len(label_encoder.classes_)} kelas")
print(f"   • Akurasi dapat dilihat dari classification report di atas")
print(f"   • Fungsi predict_anomaly() siap digunakan untuk prediksi real-time")

print("\n🔧 PENGGUNAAN:")
print("   1. Import fungsi predict_anomaly()")
print("   2. Siapkan data transaksi baru sebagai pandas Series")
print("   3. Panggil fungsi dengan parameter yang sesuai")
print("   4. Dapatkan prediksi risiko: low_risk/moderate_risk/high_risk")

print("\n⚡ TIPS OPTIMASI:")
print("   • Monitor performa model secara berkala")
print("   • Update threshold berdasarkan feedback")
print("   • Pertimbangkan ensemble methods untuk akurasi lebih tinggi")
print("   • Implementasikan real-time monitoring") 