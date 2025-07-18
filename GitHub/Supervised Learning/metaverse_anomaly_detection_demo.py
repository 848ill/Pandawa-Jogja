#!/usr/bin/env python3
"""
Skrip Demo Deteksi Anomali Transaksi Metaverse - FAST VERSION
=============================================================

Version optimized untuk presentasi dengan sampling untuk speed
Author: Muhammad Nabil Hanif
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
print("🚀 DEMO DETEKSI ANOMALI TRANSAKSI METAVERSE")
print("   FAST VERSION UNTUK PRESENTASI")
print("="*60)

# 2. Muat Data dengan SAMPLING untuk speed
print("\n1. MEMUAT DATA (SAMPLING FOR DEMO)...")
print("-" * 40)

# Sample 10K rows untuk demo yang cepat
df = pd.read_csv('metaverse_transactions_dataset.csv', 
                 parse_dates=['timestamp'], 
                 nrows=10000)  # Sample 10K untuk speed
print(f"✅ Dataset sample loaded: {df.shape[0]} baris, {df.shape[1]} kolom")

# 3. Gambaran Umum Data
print("\n2. GAMBARAN UMUM DATA")
print("-" * 30)

print(f"📊 Shape: {df.shape}")
print(f"📋 Columns: {list(df.columns)}")
print(f"🏷️ Target distribution:")
print(df['anomaly'].value_counts(normalize=True).round(3))

# 4. Feature Engineering (seperti original)
print("\n3. FEATURE ENGINEERING...")
print("-" * 30)

# Extract temporal features
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = ((df['day_of_week'] == 5) | (df['day_of_week'] == 6)).astype(int)

print("✅ Temporal features created: day_of_week, month, is_weekend")

# Drop timestamp dan features yang tidak perlu
df = df.drop(['timestamp'], axis=1)

# 5. Preprocessing
print("\n4. PREPROCESSING...")
print("-" * 30)

# Separate features dan target
X = df.drop('anomaly', axis=1)
y = df['anomaly']

# Label encoding untuk target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target classes: {label_encoder.classes_}")

# Identify numerical dan categorical columns
numerical_features = ['amount', 'hour_of_day', 'login_frequency', 
                     'session_duration', 'risk_score', 'day_of_week', 
                     'month', 'is_weekend']

categorical_features = ['transaction_type', 'location_region', 'ip_prefix',
                       'purchase_pattern', 'age_group', 'sending_address',
                       'receiving_address']

# SIMPLIFIED PREPROCESSING untuk demo
# Hanya ambil top categories untuk categorical features
for col in categorical_features:
    if col in X.columns:
        # Keep only top 5 categories, others jadi 'other'
        top_categories = X[col].value_counts().head(5).index
        X[col] = X[col].apply(lambda x: x if x in top_categories else 'other')

print("✅ Categorical features simplified untuk demo speed")

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
])

# 6. Train-Test Split
print("\n5. TRAIN-TEST SPLIT...")
print("-" * 30)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# 7. Preprocessing fit dan transform
print("\n6. APPLYING PREPROCESSING...")
print("-" * 30)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"✅ Processed training shape: {X_train_processed.shape}")
print(f"✅ Processed test shape: {X_test_processed.shape}")

# 8. Class Distribution sebelum SMOTE
print("\n7. CLASS DISTRIBUTION SEBELUM SMOTE:")
print("-" * 40)
class_counts = Counter(y_train)
total = len(y_train)

for class_label, count in class_counts.items():
    class_name = label_encoder.inverse_transform([class_label])[0]
    percentage = (count / total) * 100
    print(f"  {class_name} (label {class_label}): {count} sampel ({percentage:.1f}%)")

# 9. SMOTENC Application (dengan optimization)
print("\n8. APPLYING SMOTENC...")
print("-" * 30)

# Find categorical feature indices
categorical_indices = []
feature_names = (numerical_features + 
                [f"{col}_{val}" for col in categorical_features 
                 for val in preprocessor.named_transformers_['cat']
                 .get_feature_names_out() if col in val])

# Simplified: assume categorical features start after numerical features
n_numerical = len(numerical_features)
categorical_indices = list(range(n_numerical, X_train_processed.shape[1]))

print(f"📊 Categorical indices: {len(categorical_indices)} features")

# Apply SMOTENC dengan reduced complexity
smote_nc = SMOTENC(categorical_features=categorical_indices, 
                   random_state=42,
                   k_neighbors=3)  # Reduce neighbors untuk speed

try:
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train_processed, y_train)
    print("✅ SMOTENC applied successfully!")
    
    # Show distribution after SMOTE
    print("\n9. CLASS DISTRIBUTION SETELAH SMOTE:")
    print("-" * 40)
    class_counts_resampled = Counter(y_train_resampled)
    total_resampled = len(y_train_resampled)

    for class_label, count in class_counts_resampled.items():
        class_name = label_encoder.inverse_transform([class_label])[0]
        percentage = (count / total_resampled) * 100
        print(f"  {class_name} (label {class_label}): {count} sampel ({percentage:.1f}%)")

except Exception as e:
    print(f"⚠️ SMOTENC failed: {e}")
    print("📋 Using original unbalanced data for demo...")
    X_train_resampled, y_train_resampled = X_train_processed, y_train

# 10. Model Training
print("\n10. TRAINING RANDOM FOREST MODEL...")
print("-" * 40)

rf_model = RandomForestClassifier(
    n_estimators=50,  # Reduced untuk demo speed
    random_state=42,
    n_jobs=-1  # Use all cores
)

rf_model.fit(X_train_resampled, y_train_resampled)
print("✅ Random Forest model trained!")

# 11. Predictions dan Evaluation
print("\n11. MODEL EVALUATION...")
print("-" * 30)

y_pred = rf_model.predict(X_test_processed)

# Classification report
print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, 
                          target_names=label_encoder.classes_))

# Confusion Matrix
print("\n📊 CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance (top 10)
print("\n🎯 TOP 10 FEATURE IMPORTANCE:")
feature_importance = rf_model.feature_importances_
# Create simple feature names
feature_names_simple = numerical_features + [f"cat_{i}" for i in range(len(feature_importance) - len(numerical_features))]
importance_df = pd.DataFrame({
    'feature': feature_names_simple[:len(feature_importance)],
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(10)

for idx, row in importance_df.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# 12. Demo Predictions
print("\n12. DEMO PREDICTIONS...")
print("-" * 30)

# Create sample transactions for demo
demo_transactions = [
    {
        'amount': 150.0, 'hour_of_day': 14, 'login_frequency': 8,
        'session_duration': 120, 'risk_score': 15.0, 'day_of_week': 1,
        'month': 6, 'is_weekend': 0, 'transaction_type': 'transfer',
        'location_region': 'Europe', 'ip_prefix': '192.168',
        'purchase_pattern': 'focused', 'age_group': 'established',
        'sending_address': '0x123', 'receiving_address': '0x456'
    },
    {
        'amount': 1500.0, 'hour_of_day': 3, 'login_frequency': 1,
        'session_duration': 15, 'risk_score': 95.0, 'day_of_week': 6,
        'month': 12, 'is_weekend': 1, 'transaction_type': 'phishing',
        'location_region': 'Asia', 'ip_prefix': '10.0',
        'purchase_pattern': 'random', 'age_group': 'new',
        'sending_address': '0x999', 'receiving_address': '0x000'
    }
]

for i, transaction in enumerate(demo_transactions, 1):
    # Create DataFrame
    demo_df = pd.DataFrame([transaction])
    
    # Apply same preprocessing
    demo_processed = preprocessor.transform(demo_df)
    
    # Predict
    prediction = rf_model.predict(demo_processed)[0]
    probability = rf_model.predict_proba(demo_processed)[0]
    
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    
    print(f"\n🎯 DEMO PREDICTION {i}:")
    print(f"   Transaction: {transaction['transaction_type']} €{transaction['amount']}")
    print(f"   Risk Score: {transaction['risk_score']}")
    print(f"   📊 Predicted: {predicted_class}")
    print(f"   📊 Confidence: {max(probability):.2f}")

print("\n" + "="*60)
print("🎉 DEMO COMPLETED SUCCESSFULLY!")
print("   Ready untuk presentasi!")
print("="*60) 