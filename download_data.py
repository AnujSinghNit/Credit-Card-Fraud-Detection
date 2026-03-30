import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

def download_and_save_data():
    print("Fetching the real Credit Card Fraud Detection dataset from OpenML (ID: 1597)...")
    print("This may take a few minutes as the dataset is around 150MB.")
    
    # Fetching dataset from OpenML (version 1)
    # The 'creditcard' dataset on OpenML is the exact same one from Kaggle
    data = fetch_openml('creditcard', version=1, as_frame=True, parser='auto')
    
    df = data.frame
    
    # OpenML's target class is named 'Class' and is categorical ("0", "1")
    # Convert it to integer (0 or 1)
    if 'Class' in df.columns:
        df['Class'] = pd.to_numeric(df['Class'])
    
    # Rename columns if necessary (OpenML has 'Time' and 'Amount' but they might be lowercase or have quotes)
    # It usually has the same names: Time, V1-V28, Amount, Class
    
    print(f"Dataset completely downloaded! Total Original Shape: {df.shape}")
    
    # Performing 80-20 Train/Test split as per requirements
    print("Splitting into Train (80%) and Test (20%) datasets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])
    
    # Saving to CSV
    print("Saving train.csv...")
    train_df.to_csv("train.csv", index=False)
    
    print("Saving test.csv...")
    test_df.to_csv("test.csv", index=False)
    
    print("\n✅ Setup Complete! 'train.csv' and 'test.csv' have been successfully generated from the real dataset.")

if __name__ == "__main__":
    download_and_save_data()
