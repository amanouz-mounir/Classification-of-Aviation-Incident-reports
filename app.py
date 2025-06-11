import streamlit as st
import numpy as np
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sklearn
import lightgbm
import xgboost
import shap
import matplotlib.pyplot as plt

# Check TensorFlow availability
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Spécifie un répertoire pour télécharger les ressources nltk
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model_ML = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Téléchargement des ressources nécessaires
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('wordnet')




def clean_text(text):

    # 1. Supprimer les balises et valeurs inutiles comme [Missing Value] (non significatif)
    text = re.sub(r'\[Missing Value\]', '', text)


    # 2. Supprimer les caractères spéciaux, ponctuation et chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 3. Tokenisation : Diviser le texte en mots individuels (tokens)
    tokens = word_tokenize(text)

    # 4. Supprimer les mots vides (stopwords) : Les mots comme "the", "and" n'ont pas d'importance pour l'analyse
    stop_words = set(stopwords.words('english'))  # Liste de mots vides en anglais
    tokens = [word for word in tokens if word not in stop_words]


    # 5. Lemmatisation : Réduire les mots à leur forme de base (par exemple "running" devient "run")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]


    # 6. Rejoindre les tokens pour reformer un texte nettoyé
    cleaned_text = ' '.join(tokens)
    # 7. Convertir le texte nettoyé en minuscules
    cleaned_text = cleaned_text.lower()

    return cleaned_text


def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT

    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.

    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids

    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids


    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors

def get_bert_sentence_embedding(tokens_tensor, segments_tensors, model, max_length=512):

    device = tokens_tensor.device  # Ensure all tensors are on the same device

    all_token_embeddings = []

    # Ensure tensors have correct shape
    if tokens_tensor.dim() == 1:
        tokens_tensor = tokens_tensor.unsqueeze(0)
    if segments_tensors.dim() == 1:
        segments_tensors = segments_tensors.unsqueeze(0)

    if tokens_tensor.size(1) > max_length:
        num_chunks = (tokens_tensor.size(1) + max_length - 1) // max_length

        for i in range(num_chunks):
            start_idx = i * max_length
            end_idx = min((i + 1) * max_length, tokens_tensor.size(1))

            chunk_tokens = tokens_tensor[:, start_idx:end_idx]
            chunk_segments = segments_tensors[:, start_idx:end_idx]

            chunk_tokens = chunk_tokens.to(device)
            chunk_segments = chunk_segments.to(device)

            with torch.no_grad():
                outputs = model(chunk_tokens, token_type_ids=chunk_segments, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                token_embeddings = hidden_states.squeeze(0)

            all_token_embeddings.append(token_embeddings.mean(dim=0))

        sentence_embedding = torch.mean(torch.stack(all_token_embeddings), dim=0)

    else:
        tokens_tensor = tokens_tensor.to(device)
        segments_tensors = segments_tensors.to(device)

        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            token_embeddings = hidden_states.squeeze(0)

        sentence_embedding = torch.mean(token_embeddings, dim=0)

    return sentence_embedding.cpu().numpy()  # Move the result back to CPU before converting to numpy

def embedding(text) :
  tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
  sentence_embedding = get_bert_sentence_embedding(tokens_tensor, segments_tensors, bert_model_ML)
  return sentence_embedding


def upload_model_tokenizer(model_name): 
    """Load the model from the specified file."""
    if model_name=="Finetuning BERT":
        model = AutoModelForSequenceClassification.from_pretrained("Models/monmodele")
        tokenizer = AutoTokenizer.from_pretrained("Models/monmodele")
        return model, tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if model_name=="Xgboost":
        return joblib.load('Models/xgboost_model.joblib'), tokenizer
    elif model_name=="LGBM":
        return joblib.load('Models/lgbm_model.joblib'), tokenizer
    elif model_name=="RNN":
        if TENSORFLOW_AVAILABLE:
            return tf.keras.models.load_model('Models/RNN_model.h5', custom_objects={'loss': focal_loss()}), tokenizer
        else:
            return None, tokenizer
    elif model_name=="MAPIE + Forest":
        return joblib.load('Models/MAPIE_forest_model.pkl'), tokenizer
    else: 
        return("error: model not found"), tokenizer

def predict(indice, model_name, df_test):
    """Predict the sentiment of the text using the specified model."""
    model, tokenizer = upload_model_tokenizer(model_name)
    text = df_test.iloc[indice]['TEXT']
    text_cleaned = clean_text(text)
    if model_name=="Finetuning BERT":
        text_embedding = embedding(text_cleaned)
        inputs = tokenizer(text_cleaned, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(logits.squeeze())
        probabilities = probabilities.tolist()
        predictions = [int(prob > 0.5) for prob in probabilities]
        return predictions, probabilities, text_embedding, tokenizer, model
    else:
        if model_name=="LGBM":
            text_embedding = embedding(text_cleaned)
            probabilities = [prob[0][1] for prob in model.predict_proba([text_embedding])]
            probabilities = [float(prob) for prob in probabilities]
            predictions = [int(prob > 0.5) for prob in probabilities]
            return predictions, probabilities, text_embedding, tokenizer, model
        
        elif model_name=="Xgboost":
            text_embedding = embedding(text_cleaned)

            # Prédiction des probabilités et classes
            probabilities = model.predict_proba([text_embedding])[0]  # directement le 1er élément
            predictions = [int(p > 0.5) for p in probabilities]
            return predictions, probabilities.tolist(), text_embedding, tokenizer, model

        elif model_name=="RNN":
            if not TENSORFLOW_AVAILABLE:
                dummy_predictions = [0] * 22  
                dummy_probabilities = [0.0] * 22
                text_embedding = embedding(text_cleaned)
                return dummy_predictions, dummy_probabilities, text_embedding, tokenizer, model
            text_embedding = embedding(text_cleaned)
            probabilities = model.predict(np.array([text_embedding]))
            probabilities = probabilities[0] if probabilities.ndim > 1 else probabilities
            probabilities = np.array(probabilities, dtype=float)
            predictions = (probabilities > 0.5).astype(int)
            return predictions.tolist(), probabilities.tolist(), text_embedding, tokenizer, model
        
        elif model_name=="MAPIE + Forest":
            text_embedding = embedding(text_cleaned)
            probabilities = model.predict([text_embedding])
            predictions=np.zeros(probabilities.shape)
            predictions[probabilities>0.5]=1
            return predictions, probabilities

# Define focal_loss for custom_objects when loading the RNN model
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.where(tf.equal(y_true, 1),
                                alpha * tf.pow(1 - pt, gamma),
                                (1 - alpha) * tf.pow(1 - pt, gamma))
        bce = -tf.math.log(pt)
        return tf.reduce_mean(focal_weight * bce)
    return loss