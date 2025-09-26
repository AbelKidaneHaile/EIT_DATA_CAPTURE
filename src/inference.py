import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
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
