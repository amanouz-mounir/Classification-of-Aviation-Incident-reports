# Flight Reports Classification Demo

An interactive Streamlit application for classifying flight reports into multiple categories using AI models including transformers, LightGBM, and XGBoost.

## Features

- **Multi-model Classification**: Support for transformer models, LightGBM, and XGBoost
- **Interactive Demo**: Web-based interface for real-time text classification
- **Model Interpretability**: SHAP integration for model explainability
- **Text Preprocessing**: Specific text cleaning and preprocessing pipeline

## Prerequisites

- Python 3.11 or lower (NECESSARY FOR TENSORFLOW TO WORK!!!!!!)
- pip package manager

## Installation

### 1. Download the Project

Navigate to the project directory:
```bash
cd "/home/joker/Downloads/new cassiopee/ApplicationWeb"
```

### 2. Create a Virtual Environment with Python 3.11 (Required)

**Important**: This application requires Python 3.11 or lower for TensorFlow compatibility.

```bash
# Create virtual environment with Python 3.11 specifically
python3.11 -m venv venv

# If python3.11 is not available, you can also try:
# python3 -m venv venv  (if your system default is 3.11)
# OR install Python 3.11 first if not available on your system

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify Python version (should show 3.11.x)
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
ApplicationWeb/
├── main.py              # Main Streamlit application
├── app.py               # Core application logic and utilities
├── requirements.txt     # Python dependencies
├── README.md           # This file
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
- **MAPIE**: Not implemented due to compatibility issues with the model save

