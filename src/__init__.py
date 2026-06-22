"""Bank Customer Churn Prediction - Source Package"""

__version__ = "1.0.0"
__author__ = "Ajay Pratap Singh Hada"
__email__ = "ajaypratapsinghhada25@gmail.com"

from src.data_preprocessing import load_data, clean_data, split_data, scale_features
from src.feature_engineering import encode_categoricals, create_features
from src.model_training import train_models
from src.model_evaluation import evaluate, save_best
from src.prediction import load_model, predict

__all__ = [
    "load_data",
    "clean_data",
    "split_data",
    "scale_features",
    "encode_categoricals",
    "create_features",
    "train_models",
    "evaluate",
    "save_best",
    "load_model",
    "predict",
]
