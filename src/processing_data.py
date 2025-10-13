import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

def process_data(input_path, output_folder="data/processed", model_dir = 'models'):
    """
    Xử lý dữ liệu: mã hóa, chuẩn hóa, chia train-test, và lưu encoder/scaler.
    """

    # --- 1️⃣ Đọc dữ liệu ---
    df = pd.read_csv(input_path)

    # --- 2️⃣ Loại bỏ các hàng RF Link Quality = '0' ---
    df = df[df['RF Link Quality'] != '0'].copy()
    df.reset_index(drop=True, inplace=True)

    # --- 3️⃣ Label encoding cho RF Link Quality & Network Congestion ---
    rf_link_quality_map = {'Poor': 0, 'Moderate': 1, 'Good': 2}
    congestion_map = {'Low': 0, 'Medium': 1, 'High': 2}

    df['RF Link Quality'] = df['RF Link Quality'].map(rf_link_quality_map).astype(int)
    df['Network Congestion'] = df['Network Congestion'].map(congestion_map).astype(int)

    # --- 4️⃣ One-hot encoding cho Modulation Scheme ---
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    modulation_encoded = onehot_encoder.fit_transform(df[['Modulation Scheme']])

    # Chuyển sang DataFrame để nối lại
    modulation_encoded_df = pd.DataFrame(
        modulation_encoded,
        columns=onehot_encoder.get_feature_names_out(['Modulation Scheme'])
    )

    # Nối vào DataFrame gốc và bỏ cột cũ
    df = pd.concat([df.drop(columns=['Modulation Scheme']), modulation_encoded_df], axis=1)

    # Reset index để concat không bị lệch
    df.reset_index(drop=True, inplace=True)
    modulation_encoded_df.reset_index(drop=True, inplace=True)

    # --- 5️⃣ Chuẩn hóa Min-Max ---
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_values, columns=df.columns)

    # --- 6️⃣ Chia dữ liệu train/test ---
    train_df, test_df = train_test_split(df_scaled, test_size=0.2, random_state=42)

    # --- 7️⃣ Tạo folder lưu kết quả ---
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- 8️⃣ Lưu dữ liệu và encoder/scaler ---
    train_path = os.path.join(output_folder, "train.csv")
    test_path = os.path.join(output_folder, "test.csv")
    scaler_path = os.path.join(model_dir, "minmax_scaler.pkl")
    encoder_path = os.path.join(model_dir, "onehot_encoder.pkl")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)
    joblib.dump(onehot_encoder, encoder_path)

    print("✅ Data processing completed!")
    print(f"Train set saved to: {train_path}")
    print(f"Test set saved to: {test_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Encoder saved to: {encoder_path}")

    return train_df, test_df


if __name__ == "__main__":
    # Ví dụ chạy thử
    input_csv = "data/raw/wireless_communication_dataset.csv"  
    process_data(input_csv)
