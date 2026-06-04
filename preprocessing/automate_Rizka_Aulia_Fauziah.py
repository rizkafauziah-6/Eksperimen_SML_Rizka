import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def run_preprocessing(input_path, output_path):
    print("Memulai proses preprocessing data...")
    # 1. Load Data
    df = pd.read_csv(input_path)

    # 2. Menangani Tipe Data dan Missing Values
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df = df.dropna()

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # 3. Label Encoding untuk kolom teks (Yes/No, dll)
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # 4. Scaling untuk kolom angka
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    num_cols_exist = [col for col in num_cols if col in df.columns]
    if num_cols_exist:
        scaler = StandardScaler()
        df[num_cols_exist] = scaler.fit_transform(df[num_cols_exist])

    # 5. Simpan Data Bersih
    # Buat folder jika belum ada
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing sukses! Data disimpan di: {output_path}")

if __name__ == "__main__":
    # Karena script ini akan dijalankan dari dalam folder preprocessing
    run_preprocessing('../dataset_raw/curn-custdata.csv', 'curn_clean_preprocessing.csv')