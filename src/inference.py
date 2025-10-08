import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def predict_class(df, model, scaler):
    """Predict the class of the input data using the provided model.
    Args:
        model: Trained machine learning model with a predict method.
        df (pd.DataFrame): Input data for prediction.
    Returns:
        int: Predicted class label.
    """
    x = scaler.transform(np.array(df["Channel_C"].values).ravel().reshape(1, -1))
    y = model.predict(x)[0]

    return y

def predict_class_xgboost(df, model, scaler):
    """Predict the class of the input data using the provided model.
    Args:
        model: Trained machine learning model with a predict method.
        df (pd.DataFrame): Input data for prediction.
    Returns:
        int: Predicted class label.
    """
    x = scaler.transform(np.array(df["Channel_C"].values).ravel().reshape(1, -1))
    y = model.predict(x)[0]

    return y

def predict_class_lda_mlp(df, lda_model, mlp_model, scaler):
    """Predict the class of the input data using the provided LDA and MLP models.
    Args:
        lda_model: Trained LDA model.
        mlp_model: Trained MLP model.
        df (pd.DataFrame): Input data for prediction.
    Returns:
        int: Predicted class label.
    """
    x = scaler.transform(np.array(df["Channel_C"].values).ravel().reshape(1, -1))
    start_time = time.perf_counter()
    x_lda = lda_model.transform(x)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f'LDA transformation time: {elapsed_time*1000:.6f} miliseconds')
    
    start_time = time.perf_counter()
    y = mlp_model.predict(x_lda)[0]
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f'MLP prediction time: {elapsed_time*1000:.6f} miliseconds\n\n')
    
    return y


# def predict_1D_CNN(df, model, scaler):
    
#     # Scale the input using saved scaler
#     x_scaled = scaler.transform(df["Channel_C"].values.reshape(1, -1))
#     # Convert to tensor [batch, channels, seq_len]
#     x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(1)
#     # Forward pass
#     with torch.no_grad():
#         output = model(x_tensor)
#         predicted_class = torch.argmax(output, dim=1).item()
    
#     return predicted_class

def predict_1D_CNN(df, model, scaler):
    # Ensure model is in eval mode
    model.eval()
    
    try:
        # Scale the input using saved scaler
        x_scaled = scaler.transform(df["Channel_C"].values.reshape(1, -1))
        
        # Convert to tensor [batch, seq_len] - NO unsqueeze(1) needed!
        # The autoencoder expects [batch, input_size] format
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        
        # Forward pass
        with torch.no_grad():
            output = model(x_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        return predicted_class
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return -1  # Return invalid class on error
