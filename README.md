# Credit Card Fraud Detection with Model Comparison 💳

A full-stack, end-to-end Machine Learning web application built with **Python** and **Streamlit**. This project automatically trains and evaluates multiple classification algorithms on a highly imbalanced, real-world credit card dataset, selects the best performing model based on statistical **Recall**, and serves it through an interactive, robust dashboard.

##  Key Features

* **Authentic Data Integration**: Automatically securely downloads and segments a ~150MB 284,807-record credit card dataset strictly into training and unseen testing sets.
* **Intelligent Preprocessing**: 
  * Integrates `StandardScaler` to appropriately normalize monetary transaction values alongside PCA-transformed variables.
  * Handles monumental class imbalance securely via **SMOTE (Synthetic Minority Over-sampling Technique)** on the backend.
* **Automated Model Comparison Pipeline**: 
  * Trains **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifiers**.
  * Outputs side-by-side metric evaluations (Accuracy, Precision, Recall, F1-Score).
  * Visually maps out **Confusion Matrices** using Matplotlib Heatmaps.
* **Prioritized Metric Selection**: Automatically discards models suffering from "accuracy-fallacy" (predicting 99% accuracy by blindly guessing Legitimate) by selecting models actively maximizing true **Recall** rates. 
* **User-Friendly Dashboard**: Extremely clean Streamlit User Interface organized into distinct functional tabs.
* **One-Click Demonstrations**: Includes an integrated "Prediction Sandbox" allowing users to instantly pull random fraudulent or legitimate transactions from the underlying test dataset and see the Model correctly classify them dynamically.
* **Batch Predictions (Bulk CSV CSV)**: Effortlessly scan 10,000+ unseen transactions simultaneously by downloading a generated template and securely running them through the deployed AI.

---

##  Setup & Installation (Local Execution)

1. **Clone the Repository** and open the folder.
   ```bash
   git clone <your-repository-url>
   cd Cradit Card Fraud Detecion
   ```

2. **Install all necessary Python dependencies**:
   These packages manage the machine learning environment and the web framework.
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the Dataset & Model Pipeline**:
   *First, download the real-world dataset directly from OpenML securely to your system:*
   ```bash
   python download_data.py
   ```
   *Second, initialize the model processing pipeline. This will train all the ML models and save `model.pkl` and `scaler.pkl`:*
   ```bash
   python train_model.py
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```
   A new browser tab will immediately open visualizing your fully functioning Web UI!



## ☁️ Deployment Guide (Streamlit Cloud)

To make your web application publicly accessible without requiring users to run any training commands:

### Prep Phase
Because Streamlit Cloud spins up fresh virtual machines for free, make sure `model.pkl`, `scaler.pkl`, `model_results.json`, `app.py`, and `requirements.txt` are all forcefully pushed to your public GitHub Repository. 

### Launch Steps
1. Navigate to [Streamlit Share](https://share.streamlit.io/).
2. Click **"New App"** in the top right corner.
3. Select this repository from your GitHub dropdown. 
4. Select your main branch.
5. In the **"Main file path"**, type exactly: `app.py`.
6. Click **Deploy!**

You will now have a public, live URL running your machine-learning algorithm accessible anywhere in the world.



##  File Structure

* `app.py` - The core Streamlit application interface code. It directly reads the saved `Pickle` model for instant loading times.
* `train_model.py` - The powerhouse script utilizing Pandas, Scikit-Learn, and SMOTE to crunch all datasets and save the victor models offline.
* `download_data.py` - Utility file directly pulling down the authentic Kaggle features safely inside your code via OpenML.
* `generate_dummy_data.py` - Synthetically produces 100% identically structured mock data purely for instant local testing purposes.
* `requirements.txt` - Dictates strict infrastructure dependencies for executing the app over Streamlit Servers flawlessly.
* `*.pkl` & `*.json` files - Pre-configured compiled memory artifacts saving you from re-training large 450,000 sample Machine Learning processes every time you refresh your UI.
