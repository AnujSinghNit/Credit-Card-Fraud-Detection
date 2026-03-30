import pandas as pd
import numpy as np
import os

def create_synthetic_data(n_samples=5000, filename='dataset.csv'):
    np.random.seed(42)
    # Generate Time and Amount
    time_col = np.random.uniform(0, 172792, n_samples)
    amount_col = np.random.exponential(scale=100, size=n_samples)
    
    # Generate PCA features V1 to V28
    v_cols = {f'V{i}': np.random.normal(0, 1, n_samples) for i in range(1, 29)}
    
    # Generate labels, with slight imbalance
    class_col = np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples)
    
    df = pd.DataFrame({'Time': time_col, **v_cols, 'Amount': amount_col, 'Class': class_col})
    df.to_csv(filename, index=False)
    print(f"Created {filename} with shape {df.shape}")

if __name__ == "__main__":
    print("Generating synthetic Kaggle-style credit card data for testing...")
    create_synthetic_data(8000, 'train.csv')
    create_synthetic_data(2000, 'test.csv')
    print("Dummy dataset generation complete. You can replace train.csv and test.csv with your real Kaggle files at any time.")
