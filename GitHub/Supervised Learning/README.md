# 🚀 Deteksi Anomali Transaksi Metaverse

Sistem machine learning untuk mendeteksi anomali dalam transaksi metaverse menggunakan Random Forest Classifier dengan penanganan ketidakseimbangan kelas.

## 📋 Deskripsi

Proyek ini mengimplementasikan sistem deteksi anomali yang dapat mengklasifikasikan transaksi metaverse ke dalam tiga kategori risiko:
- **🟢 Low Risk** (`low_risk`) - Transaksi normal
- **🟡 Moderate Risk** (`moderate_risk`) - Transaksi mencurigakan
- **🔴 High Risk** (`high_risk`) - Transaksi berisiko tinggi/penipuan

## 🗂️ Struktur Proyek

```
Supervised Learning/
├── metaverse_transactions_dataset.csv    # Dataset transaksi
├── metaverse_anomaly_detection.py        # Skrip utama
├── requirements.txt                       # Dependencies
├── .cursorrules                          # Aturan deteksi anomali
└── README.md                             # Dokumentasi
```

## 🔧 Instalasi

1. **Clone repository:**
   ```bash
   git clone <repository-url>
   cd "Supervised Learning"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan skrip:**
   ```bash
   python3 metaverse_anomaly_detection.py
   ```

## 📊 Dataset

Dataset `metaverse_transactions_dataset.csv` berisi **78,600 transaksi** dengan fitur:

### Fitur Numerik:
- `amount` - Jumlah transaksi
- `hour_of_day` - Jam transaksi (0-23)
- `login_frequency` - Frekuensi login user
- `session_duration` - Durasi sesi (menit)
- `risk_score` - Skor risiko (0-100)

### Fitur Kategorikal:
- `transaction_type` - Jenis transaksi (transfer, purchase, phishing, scam)
- `location_region` - Wilayah geografis
- `ip_prefix` - Prefix alamat IP
- `purchase_pattern` - Pola pembelian (focused, random, high_value)
- `age_group` - Kategori umur user (new, established, veteran)
- `sending_address` - Alamat pengirim
- `receiving_address` - Alamat penerima

### Target Variable:
- `anomaly` - Label risiko (low_risk, moderate_risk, high_risk)

## 🤖 Model & Pipeline

### 1. Feature Engineering
- Ekstraksi `day_of_week` dan `month` dari timestamp
- Fitur binary `is_weekend`
- Normalisasi fitur numerik dengan `StandardScaler`
- One-hot encoding untuk fitur kategorikal

### 2. Penanganan Ketidakseimbangan Kelas
- Menggunakan **SMOTENC** (SMOTE untuk data mixed)
- Mengatasi dominasi kelas `low_risk`
- Meningkatkan deteksi kelas minoritas (`high_risk`, `moderate_risk`)

### 3. Model Training
- **Random Forest Classifier** dengan 100 estimators
- Train-test split 80:20 dengan stratifikasi
- Evaluasi menggunakan classification report dan confusion matrix

## 🎯 Penggunaan

### Menjalankan Analisis Lengkap:
```python
python3 metaverse_anomaly_detection.py
```

### Prediksi Transaksi Baru:
```python
import pandas as pd

# Contoh transaksi baru
new_transaction = pd.Series({
    'hour_of_day': 14,
    'amount': 150.0,
    'transaction_type': 'transfer',
    'location_region': 'Europe',
    'ip_prefix': '192.168',
    'login_frequency': 8,
    'session_duration': 120,
    'purchase_pattern': 'focused',
    'age_group': 'established',
    'risk_score': 15.0,
    'day_of_week': 1,
    'month': 6,
    'is_weekend': 0,
    'sending_address': '0x1234567890abcdef',
    'receiving_address': '0xfedcba0987654321'
})

# Prediksi (setelah menjalankan skrip utama)
prediction = predict_anomaly(
    new_transaction, 
    preprocessor, 
    rf_model, 
    label_encoder, 
    original_columns
)
print(f"Prediksi risiko: {prediction}")
```

## 📈 Contoh Output

```
============================================================
DETEKSI ANOMALI TRANSAKSI METAVERSE
============================================================

Dataset berhasil dimuat: 78600 baris, 14 kolom

📊 Distribusi kelas SEBELUM SMOTE:
  low_risk (label 0): 62880 sampel (89.8%)
  moderate_risk (label 1): 5030 sampel (7.2%)
  high_risk (label 2): 2090 sampel (3.0%)

📊 Distribusi kelas SETELAH SMOTE:
  low_risk (label 0): 62880 sampel (33.3%)
  moderate_risk (label 1): 62880 sampel (33.3%)
  high_risk (label 2): 62880 sampel (33.3%)

🔍 CONTOH PREDIKSI:
📋 Input: Transfer €150, established user, risk_score=15.0
🎯 Prediksi: low_risk

📋 Input: Phishing €1500, new user, risk_score=95.0
🎯 Prediksi: high_risk
```

## ⚙️ Aturan Deteksi (.cursorrules)

Proyek ini menggunakan sistem aturan yang terdefinisi dalam `.cursorrules`:

### Kriteria Risiko Tinggi:
- `transaction_type` = 'phishing' atau 'scam'
- `risk_score` ≥ 90
- `anomaly` = 'high_risk'

### Kriteria Risiko Sedang:
- `risk_score` antara 50-89.99
- User baru dengan pola pembelian acak
- Login frequency rendah

### Kriteria Risiko Geografis:
- Wilayah Asia dan Africa (risiko lebih tinggi)
- `risk_score` ≥ 40

## 🎯 Metrik Evaluasi

Model dievaluasi dengan fokus pada:
- **Precision, Recall, F1-Score** untuk setiap kelas
- **Confusion Matrix** untuk analisis kesalahan klasifikasi
- **Prioritas tinggi** pada deteksi kelas `high_risk` dan `moderate_risk`

## 🚀 Optimasi & Pengembangan

### Saran Peningkatan:
1. **Ensemble Methods** - Kombinasikan multiple models
2. **Hyperparameter Tuning** - GridSearch/RandomSearch
3. **Feature Selection** - Analisis importance fitur
4. **Real-time Monitoring** - Implementasi streaming detection
5. **Model Retraining** - Update berkala dengan data baru

### Monitoring:
- Track model performance over time
- Update threshold berdasarkan feedback
- Monitor distribusi data untuk concept drift

## 📚 Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
```

## 👤 Author

**Muhammad Nabil Hanif**
- Project: Supervised Learning - Anomaly Detection
- Dataset: Metaverse Transactions

## 📄 License

This project is for educational and research purposes.

---

**⚠️ Important Note**: Skor F1 untuk kelas 'high_risk' dan 'moderate_risk' adalah metrik kinerja yang paling kritis karena ketidakseimbangan kelas. Focus utama adalah mendeteksi transaksi berisiko tinggi dan sedang untuk mencegah penipuan dalam ecosystem metaverse. 