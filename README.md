# Flight Reports Classification Project

An interactive Streamlit application for classifying flight reports into multiple categories using AI models including transformers, LightGBM, and XGBoost.

## Features

- **Multi-model Classification**: Support for transformer models, LightGBM, and XGBoost
- **Interactive Demo**: Web-based interface for real-time text classification
- **Model Interpretability**: SHAP integration for model explainability
- **Text Preprocessing**: Specific text cleaning and preprocessing pipeline

## Installation

### 1. Download the Project

Navigate to the project directory:
```bash
cd "~/Repository/ApplicationWeb"
```

### 2. Create a Virtual Environment with Python 3.12.3


```bash
# Create virtual environment with Python 3.12.3
python3 -m venv venv

# If python3 is not available, you can also try:
# python3 -m venv venv  (if your system default is 3.12)
# OR install Python 3.12.3 first if not available on your system

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify Python version 
python --version
```



### 3. Install Dependencies

Install all required packages using the requirements file:

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data (if needed)

If you encounter NLTK-related errors, download the required NLTK data:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Running the Application

### Start the Streamlit App

Run the main application:

```bash
streamlit run main.py
```

## Usage

1. **Open the Application**: Navigate to the provided URL in your web browser
2. **Select text index**: Choose an index from the available ones to select a text
3. **Select Model**: Choose from available AI models (Transformer, LightGBM, XGBoost)
4. **Classify Text**: Input text or use uploaded data for classification
5. **View Results**: Analyze classification results and model explanations

## Project Structure

```
|-README.md
|-ApplicationWeb/
└── ├──main.py              # Main Streamlit application
    ├── app.py               # Core application logic and utilities
    ├── requirements.txt     # Python dependencies
    ├── TestTruth_Brut.csv  # Sample data file
    ├── Models/             # Pre-trained models directory
    │   ├── lgbm_model.joblib
    │   ├── RNN_model.h5
    │   ├── xgboost_model.joblib
    │   └── monmodele/      # Transformer model files
    └── Images/             # Application assets
        ├── logo3.png
        ├── clean_text.png
        └── text.png
```

## Models

The application supports multiple AI models:

- **Transformer Models**: Advanced neural networks for text classification (BERT Model)
- **LightGBM**: Gradient boosting framework for structured data
- **XGBoost**: Extreme gradient boosting for high-performance classification
- **RNN**: Recurrent neural network for sequence processing
- **MAPIE + Random Forest**: Uncertainty quantification method combined with ensemble learning for robust predictions

## 🔍 About the BERT Model

The BERT model used in this project is **not included in this GitHub repository** due to its large size.

Similarly, the **MAPIE + Random Forest**, and **RNN** models are not included here due to compatibility issues with the model save.

📦 If you would like to obtain any of these models, please feel free to contact me directly by email.

Thank you for your understanding!


