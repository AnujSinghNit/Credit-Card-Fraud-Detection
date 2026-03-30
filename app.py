import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Helper function to safely load resources
@st.cache_resource
def load_resources():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_results.json', 'r') as f:
            results = json.load(f)
        return model, scaler, results
    except Exception as e:
        return None, None, None

def render_comparison(results):
    st.header("📊 Model Comparison Results")
    st.write("Below is the performance comparison of the trained models based on the unseen test dataset.")
    
    # Exclude non-model keys
    models_data = {k: v for k, v in results.items() if k not in ['Best_Model', 'Features']}
    
    df_results = pd.DataFrame(models_data).T.drop(columns=['Confusion_Matrix'])
    df_results = df_results.apply(pd.to_numeric)
    
    # Highlight the best model's row
    best_model = results.get('Best_Model', '')
    
    st.dataframe(df_results.style.apply(lambda x: ['background: #1e3d59' if x.name == best_model else '' for i in x], axis=1), use_container_width=True)
    
    st.success(f"🏆 **Best Performing Model**: {best_model} (Automatically Selected due to highest Recall)")

    st.subheader("Performance Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    df_results[['F1-score', 'Recall']].plot(kind='bar', ax=ax, color=['#ff6e40', '#1e3d59'])
    plt.xticks(rotation=45)
    plt.ylabel('Score')
    st.pyplot(fig)


def render_confusion_matrices(results):
    st.header("🧩 Confusion Matrices")
    st.write("Visualize the True Positives, True Negatives, False Positives, and False Negatives for each model.")
    
    models_data = {k: v for k, v in results.items() if k not in ['Best_Model', 'Features']}
    
    cols = st.columns(len(models_data))
    
    for i, (name, metrics) in enumerate(models_data.items()):
        with cols[i]:
            st.subheader(name)
            cm = np.array(metrics['Confusion_Matrix'])
            
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Legit (0)', 'Fraud (1)'])
            ax.set_yticklabels(['Legit (0)', 'Fraud (1)'])
            st.pyplot(fig)

def render_prediction_system(model, scaler, features):
    st.header("🛡️ Fraud Detection Input Form")
    st.write("Enter transaction details below to check for potential credit card fraud, or upload a CSV containing multiple transactions.")

    pred_mode = st.radio("Prediction Mode", ["Single Transaction", "Bulk Upload (CSV)"], horizontal=True)
    
    if pred_mode == "Single Transaction":
        st.write("Click a button below to pull a random transaction from the secure dataset and scan it in real-time. No manual data entry required!")
        
        col1, col2 = st.columns(2)
        with col1:
            test_legit = st.button("✅ Scan Random Legitimate Transaction", use_container_width=True)
        with col2:
            test_fraud = st.button("🚨 Scan Random Fraudulent Transaction", use_container_width=True)
            
        transaction_to_test = None
        
        if test_legit or test_fraud:
            try:
                # Optimally read CSV without caching huge file just to get 1 row
                # We can just randomly sample
                df_test = pd.read_csv("test.csv")
                
                if test_legit:
                    sample_row = df_test[df_test['Class'] == 0].sample(1, random_state=np.random.randint(0, 10000))
                else:
                    sample_row = df_test[df_test['Class'] == 1].sample(1, random_state=np.random.randint(0, 10000))
                    
                transaction_to_test = sample_row.drop(columns=['Class'])
            except Exception as e:
                st.error("Could not load dataset. Make sure test.csv exists.")
                
        if transaction_to_test is not None:
            st.write("### Extracted Transaction Variables")
            st.dataframe(transaction_to_test, hide_index=True)
            
            # Formally Predict
            input_scaled = scaler.transform(transaction_to_test)
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            confidence = max(proba) * 100
            
            st.markdown("---")
            if prediction == 1:
                st.error(f"### ❌ Fraudulent Transaction Detected! (Decision Confidence: {confidence:.2f}%)")
            else:
                st.success(f"### ✅ Legitimate Transaction. (Decision Confidence: {confidence:.2f}%)")
                    
    else: # Bulk Upload
        st.write("Provide a CSV with the exact 30 features to predict massive amounts of transactions at once.")
        # Create sample template
        sample_df = pd.DataFrame(columns=features)
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Sample CSV Template", data=sample_csv, file_name="sample_transactions.csv", mime='text/csv')
        
        uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            
            # Check if all required features exist
            missing_cols = [col for col in features if col not in df_upload.columns]
            if missing_cols:
                st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}")
            else:
                # Prepare and predict
                X_batch = df_upload[features].copy()
                # Impute missing if any in upload
                X_batch = X_batch.fillna(0)
                
                X_batch_scaled = scaler.transform(X_batch)
                
                predictions = model.predict(X_batch_scaled)
                confidences = np.max(model.predict_proba(X_batch_scaled), axis=1) * 100
                
                df_upload['Prediction'] = ['Fraud' if p == 1 else 'Legit' for p in predictions]
                df_upload['Confidence (%)'] = confidences
                
                st.write("### Batch Prediction Results")
                
                # Highlight Fraud rows
                def highlight_fraud(row):
                    if row['Prediction'] == 'Fraud':
                        return ['background-color: #ffcccc'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(df_upload.style.apply(highlight_fraud, axis=1), height=400, use_container_width=True)
                
                fraud_count = (df_upload['Prediction'] == 'Fraud').sum()
                st.warning(f"Found {fraud_count} potentially fraudulent transactions out of {len(df_upload)} total.")


def main():
    st.markdown("""
        <style>
        /* 1. App Background Color (Interactive Animated Gradient) */
        .stApp {
            background: linear-gradient(-45deg, #E3F2FD, #FFFFFF, #BBDEFB, #F4F6F9);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* 2. Main Content Card Wrapper (Glassmorphism) */
        .block-container {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(16px);
            padding: 2.5rem !important;
            border-radius: 24px;
            box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.1);
            margin-top: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.6);
        }

        /* 3. Navigation Bar (Tabs) -> Converted into Distinct Pills */
        button[data-baseweb="tab"] {
            background-color: transparent !important;
            border: none !important;
        }
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
            background-color: #F1F5F9;
            padding: 10px 20px;
            border-radius: 50px;
            color: #475569;
            font-weight: 700;
            margin: 0;
            transition: all 0.2s ease;
        }
        button[data-baseweb="tab"][aria-selected="true"] > div[data-testid="stMarkdownContainer"] > p {
            background-color: #2563EB;
            color: white;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
        }
        button[data-baseweb="tab"]:hover > div[data-testid="stMarkdownContainer"] > p {
            background-color: #DBEAFE;
            color: #1D4ED8;
        }

        /* 4. Button Styling & Hover Effects (Ultra Interactive) */
        .stButton>button {
            border-radius: 12px;
            height: 3.2rem;
            font-size: 16px;
            font-weight: 700;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            background-color: #FFFFFF;
            border: 2px solid #E2E8F0;
            color: #1E293B;
        }
        .stButton>button:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.2);
            border-color: #2563EB;
            color: #2563EB;
        }
        .stButton>button:active {
            transform: translateY(0px);
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.1);
        }

        /* 5. The Header Gradient (Animated) */
        .custom-header {
            background: linear-gradient(270deg, #2563EB, #38BDF8, #818CF8, #2563EB);
            background-size: 300% 300%;
            animation: headerFlow 8s ease infinite;
            padding: 30px;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        @keyframes headerFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        </style>
        <div class="custom-header">
            <h1 style="color: white; margin: 0; font-size: 38px; font-weight: 800;">💳 Credit Card Fraud Detection</h1>
            <p style="margin: 0; font-size: 18px; opacity: 0.95; padding-top: 5px;">Enterprise-Grade Machine Learning Predictor</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Try loading resources
    model, scaler, results = load_resources()
    
    if not model or not results:
        st.warning("⚠️ Model data not found. Please assure `model.pkl`, `scaler.pkl`, and `model_results.json` exist. (Run `train_model.py` to generate them).")
        st.stop()
        
    tab1, tab2, tab3 = st.tabs(["🛡️ Fraud Prediction", "📊 Model Comparison", "🧩 Confusion Matrices"])
    
    with tab1:
        render_prediction_system(model, scaler, results.get('Features', []))
        
    with tab2:
        render_comparison(results)
        
    with tab3:
        render_confusion_matrices(results)

if __name__ == "__main__":
    main()
