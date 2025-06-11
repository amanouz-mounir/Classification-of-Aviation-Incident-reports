import streamlit as st
import streamlit.components.v1 as components
import pandas as pd # For managing the dataframe of texts
from transformers import pipeline, Pipeline  
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import re
from app import clean_text, upload_model_tokenizer, predict, embedding
import shap
import numpy as np
import base64




# --- 0. Page Configuration (Optional but good practice) ---
st.set_page_config(page_title="Interactive Demo: Classifying Flight Reports into Multiple Categories with AI", layout="wide")

# --- 1. Global Variables / Data Loading (Placeholder) ---
# This would be your actual data. For the demo, we can create a mock one.
@st.cache_data # Cache data for performance
def load_data():
    # Load the model
    df_test= pd.read_csv('TestTruth_Brut.csv')
    return df_test

df_texts = load_data()

# Fixed list of 22 aviation problem labels
CANDIDATE_LABELS = [ f"Problem {k}" for k in range(22) ]# Replace with actual labels

MAX_INDEX = len(df_texts) - 1

# Placeholder for model functions (replace with your actual logic)
def placeholder_clean_text(text):
    return clean_text(text)

# Chargement de l'image locale en base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_bin_file("Images/logo3.png")
logo_text = get_base64_of_bin_file("Images/text.png")
logo_clean = get_base64_of_bin_file("Images/clean_text.png")

# Titre avec logo à gauche mais texte centré horizontalement
st.markdown(
    f"""
    <style>
        .header-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            padding: 20px 0 10px 0;
            flex-wrap: wrap;
        }}
        .header-logo {{
            height: 100px;
            width: auto;
        }}
        .header-title {{
            font-size: 38px;
            font-weight: bold;
            color: #f0f4f8;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
            flex: 1;
        }}
        .section-divider {{
            border-top: 2px solid #1f3b60;
            margin-top: 10px;
            margin-bottom: 20px;
        }}
    </style>

    <div class="header-container">
        <img src="data:image/jpg;base64,{logo_base64}" class="header-logo">
        <div class="header-title">AI-based Classification of Aviation Incident Reports by Problem Type</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

with st.expander(
    "Site Description - click to expand/collapse",
    expanded=False,
):
    st.markdown("""
    This site allows you to select a text from a test dataset containing **7077 reports**.
    The goal is to classify this text into **22 different labels**.

    This is a **multi-label classification** (a sample can belong to multiple labels).

    How to use the site:

    1. Select the report index in the Test.txt file between **0 and 7076**.
    2. Choose the classification model.
    3. Click on **Test and Predict**.

    This operation will trigger the prediction, displaying:
    - The cleaned text given as input to the model
    - Model performance metrics.
    - The model's prediction.
    - The ground truth for comparison to check prediction accuracy.
    - SHAP explainability to understand the model's decision.

    You can collapse or expand this description panel anytime by clicking on the title.
    """, unsafe_allow_html=False)



# --- 3. Sidebar for Inputs (or main area, as per your sketch) ---
# The sketch seems to have inputs at the top, so we'll use the main area.

# --- 4. Input Controls (Top Area) ---
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    # Text Index Selection (as per "indice du texte")
    # Using a number input is more user-friendly than 4 separate boxes for a single number
    selected_index = st.number_input(
        f"Select the report index from the test file (0 to {MAX_INDEX})",
        min_value=0,
        max_value=MAX_INDEX,
        value=0, # Default value
        step=1,
        key="text_index"
    )

with col2:
    # Model Selection (as per "modele" dropdown)
    model_options = [
        "Xgboost", "LGBM", "MAPIE + Forest", "Finetuning BERT", "RNN",
 ]
    selected_model_name = st.selectbox(
        "Choose a Model",
        options=model_options,
        key="model_select"
    )

with col3:
    # "OK" / "Prédire" Button
    st.write("") # For spacing
    st.write("") # For spacing
    predict_button = st.button("👁️ Show & Predict", use_container_width=True)

st.markdown("---")

# --- 5. Display Area (Auto for Original Text) ---

# --- 5. Display Area (Auto for Original Text Only) ---
if selected_index is not None and selected_index in df_texts.index:
    original_text = df_texts.loc[selected_index, 'TEXT']
    
    # Affichage des deux colonnes pour garder l'alignement
    text_col1, text_col2 = st.columns(2)
    
    with text_col1:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 10px;">
                <img src="data:image/png;base64,{logo_text}" alt="icon" style="height: 55px;">
                <h3 style="margin: 0; color: #f0f4f8;">Original Text</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.text_area("", value=original_text, height=200, disabled=True, key="original_text_area_preview")
    
    with text_col2:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 10px;">
                <img src="data:image/png;base64,{logo_clean}" alt="icon" style="height: 55px;">
                <h3 style="margin: 0; color: #f0f4f8;">Text Cleaned</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if predict_button:
            # Nettoyer le texte et afficher le résultat
            cleaned_text = clean_text(original_text)
            st.text_area("", value=cleaned_text, height=200, disabled=True, key="cleaned_text_area")
        else:
            # Texte d'attente par défaut
            st.text_area(
                "",
                value="Click on 'Show & Predict' to display the cleaned text.", 
                height=200,
                disabled=True,
                key="cleaned_text_placeholder"
            )


# --- 5. Display Area (Triggered by Button) ---
if predict_button:
    st.session_state.results_displayed = True # Flag to keep results visible

    # --- 5.1. Fetch and Display Original & Cleaned Text ---
    try:
        original_text = df_texts.loc[selected_index, 'TEXT']
        true_labels_list = df_texts.loc[selected_index, df_texts.columns[:22]].tolist()
    except KeyError:
        st.error(f"Error: Index {selected_index} not found in the dataset.")
        st.stop() # Stop execution if index is invalid

    # Call your actual cleaning function
    # cleaned_text = clean_text_function(original_text) # Replace with your function
    cleaned_text = clean_text(original_text) # Using placeholder


    # --- 5.2. Load Model and Get Predictions ---
    st.markdown("## Models and Metrics")
    if selected_model_name == "MAPIE + Forest":
        st.info("MAPIE + Forest model is not yet implemented in this demo.")
        st.stop()
    elif selected_model_name == "RNN":
        st.markdown(
        """
        <style>
        .big-metric-box {
            display: inline-block;
            padding: 20px 35px;
            background-color: #0a1a2f;
            border-radius: 14px;
            border: 1px solid #1f3b60;
            color: #d0d8e8;
            font-size: 20px;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .big-metric-box:hover {
            transform: scale(1.1);
            box-shadow: 0 0 25px rgba(30, 144, 255, 0.9), 0 0 10px rgba(30, 144, 255, 0.7);
        }
        </style>
        <div style='display: flex; justify-content: center;'>
        <div class='big-metric-box'>
            <strong>F1-score :</strong> 0.81&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Precision :</strong> 0.71&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Recall :</strong> 0.58
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    elif selected_model_name == "Finetuning BERT":
        st.markdown(
    """
    <style>
        .big-metric-box {
            display: inline-block;
            padding: 20px 35px;
            background-color: #0a1a2f;
            border-radius: 14px;
            border: 1px solid #1f3b60;
            color: #d0d8e8;
            font-size: 20px;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .big-metric-box:hover {
            transform: scale(1.1);
            box-shadow: 0 0 25px rgba(30, 144, 255, 0.9), 0 0 10px rgba(30, 144, 255, 0.7);
        }
    </style>
    <div style='display: flex; justify-content: center;'>
        <div class='big-metric-box'>
            <strong>F1-score :</strong> 0.666&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Precision :</strong> 0.622&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Recall :</strong> 0.822
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    elif selected_model_name == "Xgboost":
      st.markdown(
    """
    <style>
        .big-metric-box {
            display: inline-block;
            padding: 20px 35px;
            background-color: #0a1a2f;
            border-radius: 14px;
            border: 1px solid #1f3b60;
            color: #d0d8e8;
            font-size: 20px;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .big-metric-box:hover {
            transform: scale(1.1);
            box-shadow: 0 0 25px rgba(30, 144, 255, 0.9), 0 0 10px rgba(30, 144, 255, 0.7);
        }
    </style>

    <div style='display: flex; justify-content: center;'>
        <div class='big-metric-box'>
            <strong>F1-score :</strong> 0.626&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Précision :</strong> 0.80&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Recall :</strong> 0.546
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



    
    elif selected_model_name == "LGBM":
        st.markdown(
    """
    <style>
        .big-metric-box {
            display: inline-block;
            padding: 20px 35px;
            background-color: #0a1a2f;
            border-radius: 14px;
            border: 1px solid #1f3b60;
            color: #d0d8e8;
            font-size: 20px;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .big-metric-box:hover {
            transform: scale(1.1);
            box-shadow: 0 0 25px rgba(30, 144, 255, 0.9), 0 0 10px rgba(30, 144, 255, 0.7);
        }
    </style>
    <div style='display: flex; justify-content: center;'>
        <div class='big-metric-box'>
            <strong>F1-score :</strong> 0.625&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Precision :</strong> 0.660&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Recall :</strong> 0.627
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown("---")


    # Call your actual model loading and prediction functions
    # loaded_model = load_model(selected_model_name) # Replace
    # predictions = predict_with_model(loaded_model, cleaned_text) # Replace (should return list of tuples: (label, probability))

    raw_predictions, raw_probabilities, text_embedding, tokenizer,model = predict(selected_index, selected_model_name, df_texts) # Using placeholder

    # --- 5.3. Display Predictions (with percentage and color) ---
    # Define a color scheme for probabilities
   

    st.markdown("## Prediction Results")

    cols = st.columns(5)  # 5 colonnes par ligne

#ATTENTION CONDITIONNER CELA SUR LGBM 
    for i, label in enumerate(CANDIDATE_LABELS):
        # Définir le fond : bleu nuit si prédit, gris clair sinon
        bg_color = "#0b1f3a" 
        text_color = "#ffffff" 
        border_color = "#195089" 

        icon = "✅" if raw_predictions[i] else "❌"
        bar_width = int(raw_probabilities[i] * 100)
        bar_color = "green" if raw_predictions[i] else "red"
        
        with cols[i % 5]:
            st.markdown(f"""
                <div style='
                    border: 1px solid {border_color};
                    border-radius: 10px;
                    padding: 12px;
                    background-color: {bg_color};
                    margin-bottom: 12px;
                    font-size: 14px;
                '>
                    <div style='font-weight: bold; color: {text_color};'>
                        {icon} {label}
                    </div>
                    <div style='color: {text_color}; margin-top: 4px;'>
                        {raw_probabilities[i]:.2%}
                    </div>
                    <div style='height: 8px; background: #ddd; border-radius: 4px; margin-top: 6px;'>
                        <div style='width: {bar_width}%; background: {bar_color}; height: 100%; border-radius: 4px;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)


    st.markdown("---")

    st.markdown("## Ground Truth ")

    cols = st.columns(5)  # 3 colonnes

    for i, label in enumerate(CANDIDATE_LABELS):
        is_true = true_labels_list[i] == 1  # 1 ou 0 pour ce label

        bg_color = "#4CAF50" if is_true else "#0b1f3a"  # vert ou bleu nuit
        text_color = "white" 

        with cols[i % 5]:
            st.markdown(f"""
                <div style="
                    border:1px solid #ddd;
                    background-color:{bg_color};
                    padding:12px;
                    border-radius:8px;
                    margin-bottom:10px;
                    text-align:center;
                    font-weight:bold;
                    font-size:14px;
                    color:{text_color};
                ">
                    {label}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## Explainability with SHAP Values")
    def st_shap_text(shap_values_instance):
        """Affiche un plot SHAP text() dans Streamlit avec une boîte gris clair et texte noir."""
        with st.spinner(" Displaying SHAP values..."):

            shap_html = f"""
            <head>
                {shap.getjs()}
                <style>
                    body {{
                        background-color: transparent;
                        color: black;
                    }}
                    .shap-plot-text span {{
                        color: black !important;
                    }}
                    .shap-box {{
                        background-color: #ffffff;
                        border-radius: 10px;
                        padding: 15px;
                        border: 1px solid #ccc;
                        max-height: 400px;
                        overflow-y: auto;
                        font-family: sans-serif;
                    }}
                </style>
            </head>
            <body>
                <div class="shap-box">
                    {shap.plots.text(shap_values_instance, display=False)}
                </div>
            </body>
            """
            components.html(shap_html, height=500, scrolling=False)


    # Afficher les valeurs SHAP
    if selected_model_name == "Xgboost":
       def shap_predict(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        embeddings = [embedding(text) for text in texts]
        X = np.vstack(embeddings)
        return model.predict_proba(X)  # shape (n_samples, n_classes)
       
       with st.spinner(" Calculating SHAP values..."):
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(shap_predict, masker)
        shap_values = explainer([cleaned_text], fixed_context=1)
        
    # Utilisation
       st_shap_text(shap_values)

    elif selected_model_name == "LGBM":
        def shap_predict(texts):
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()

            embeddings = [embedding(text) for text in texts]
            X = np.vstack(embeddings)
            
            probs = model.predict_proba(X)  # Pour multilabel, liste de n_classes éléments
            if isinstance(probs, list) and isinstance(probs[0], np.ndarray):
                # Cas multilabel : chaque classe a un array (n_samples, 2) avec probas [0, 1]
                # On extrait les probas "classe 1" pour chaque label
                probs = np.array([p[:, 1] for p in probs]).T  # Transpose: (n_samples, n_classes)
            return probs



        # 6. SHAP : texte → masking → prédiction
        with st.spinner(" Calculating SHAP values..."):
            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(shap_predict, masker)
            shap_values = explainer([cleaned_text], fixed_context=1)
        st_shap_text(shap_values)
  
    elif selected_model_name == "Finetuning BERT":
        def shap_predict(texts):
            import numpy as np
            # Convert ndarray en liste Python si besoin
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            elif isinstance(texts, str):
                texts = [texts]
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.sigmoid(outputs.logits)

            return probs.cpu().numpy()
        with st.spinner(" Calculating SHAP values..."):
            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(shap_predict, masker)
            shap_values = explainer([cleaned_text], fixed_context=1)        
        # Utilisation et affichage des shap_values
        st_shap_text(shap_values)

    elif selected_model_name == "RNN":
        def shap_predict(texts):
            import numpy as np
            # Convert ndarray en liste Python si besoin
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            elif isinstance(texts, str):
                texts = [texts]
            
            # Create embeddings for each text
            embeddings = [embedding(text) for text in texts]
            X = np.vstack(embeddings)
            
            # Get predictions from the RNN model
            probs = model.predict(X)
            # Handle dimensions properly
            if probs.ndim > 2:
                probs = probs.squeeze()
            elif probs.ndim == 1:
                probs = probs.reshape(1, -1)
                
            return probs
            
        with st.spinner(" Calculating SHAP values..."):
            masker = shap.maskers.Text(tokenizer)
            explainer = shap.Explainer(shap_predict, masker)
            shap_values = explainer([cleaned_text], fixed_context=1)
        # Utilisation et affichage des shap_values
        st_shap_text(shap_values)

    else:
        st.warning("SHAP values are not available for this model.")
    


# --- 6. (Optional) Footer or further explanations ---
st.markdown("---")
st.caption("Demo app for visualizing multi-label classification results.")
