# 🏦 Bank Customer Churn Prediction

A comprehensive machine learning project to predict customer churn in European banks using classification algorithms.

## 📋 Project Overview

This project analyzes bank customer data and builds predictive models to identify customers at risk of leaving the bank. The solution includes:

- **Data Exploration & Analysis**: Comprehensive EDA of customer demographics and behavior
- **Feature Engineering**: Creation of meaningful features from raw data
- **Model Training**: Multiple classification models (Logistic Regression, Random Forest, XGBoost, etc.)
- **Model Evaluation**: Rigorous evaluation using cross-validation and multiple metrics
- **Interactive Dashboard**: Streamlit-based web application for predictions

## 📁 Project Structure

```
├── data/
│   ├── raw/                          # Raw dataset (European_Bank.csv)
│   └── processed/                    # Processed/cleaned data
├── src/
│   ├── data_preprocessing.py         # Data loading, cleaning, splitting
│   ├── feature_engineering.py        # Feature encoding & creation
│   ├── model_training.py             # Model training utilities
│   ├── model_evaluation.py           # Evaluation metrics & selection
│   └── prediction.py                 # Inference on new data
├── notebooks/
│   ├── 01_Data_Understanding.ipynb   # Initial data exploration
│   ├── 02_EDA.ipynb                  # Exploratory data analysis
│   ├── 03_Feature_Engineering.ipynb  # Feature creation & selection
│   ├── 04_Model_Training.ipynb       # Model training & tuning
│   └── 05_Model_Evaluation.ipynb     # Model evaluation & comparison
├── dashboard/
│   ├── app.py                        # Streamlit dashboard
│   └── requirements.txt              # Dashboard dependencies
├── models/                           # Trained model files
├── images/                           # Visualizations & plots
└── reports/                          # Analysis reports
```

## 📊 Dataset

**Source**: European Bank Customer Dataset

**Features**:
- Customer demographics (Age, Gender, Geography)
- Account information (Balance, Tenure, Number of Products)
- Activity metrics (Credit Score, Salary, Credit Card status)
- Target: Exited (0 = Stayed, 1 = Churned)

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/imajay8225/Bank-Customer-Churn-Prediction.git
cd Bank-Customer-Churn-Prediction
pip install -r requirements.txt
```

### Run Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

### Launch Dashboard
```bash
cd dashboard
streamlit run app.py
```

## 🔧 Workflow

1. **Data Preprocessing** → Clean & normalize raw data
2. **Feature Engineering** → Create domain-specific features
3. **Exploratory Analysis** → Understand data distributions & relationships
4. **Model Training** → Train multiple classification models
5. **Model Evaluation** → Compare performance using ROC-AUC, F1, Accuracy
6. **Predictions** → Deploy best model for inference
7. **Dashboard** → Interactive web interface for predictions

## 📈 Key Metrics

- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 💻 Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Plotly** - Interactive visualizations
- **Streamlit** - Web dashboard
- **Jupyter** - Interactive notebooks

## 📝 Usage

### Making Predictions

```python
from src.prediction import load_model, predict
import pandas as pd

# Load best model
model = load_model("models/best_model.pkl")

# Load new customer data
customers = pd.read_csv("new_customers.csv")

# Get predictions
results = predict(model, customers)
print(results[["CustomerId", "Churn_Probability", "Churn_Prediction"]])
```

### Training Custom Models

```python
from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import encode_categoricals, create_features
from src.model_training import train_models
from src.model_evaluation import evaluate, save_best

# Load & preprocess
df = load_data("data/raw/European_Bank.csv")
df = clean_data(df)
X_train, X_test, y_train, y_test = split_data(df)

# Feature engineering
X_train = encode_categoricals(X_train)
X_train = create_features(X_train)

# Train models
models = train_models(X_train, y_train)

# Evaluate & save best
results = [evaluate(model, X_test, y_test) for model in models.values()]
best = save_best(models, results)
```

## 📊 Expected Results

- **Best Model ROC-AUC**: ~0.84-0.88
- **Accuracy**: ~80-85%
- **Precision**: ~60-70%
- **Recall**: ~50-60%

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ajay Pratap Singh Hada**
- GitHub: [@imajay8225](https://github.com/imajay8225)
- Email: ajaypratapsinghhada25@gmail.com

## 🙏 Acknowledgments

- Dataset sourced from European Bank customer records
- Inspired by industry best practices in churn prediction
- Special thanks to the open-source ML community

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ML Classification Best Practices](https://machinelearningmastery.com/)

---

**Last Updated**: June 2026
**Status**: ✅ Production Ready
