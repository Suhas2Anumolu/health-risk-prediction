
# Health Risk Prediction using Machine Learning

This project applies supervised machine learning techniques to predict health risk levels (Low, Medium, High) based on individual health-related features. It demonstrates a complete pipeline, including preprocessing, model training, evaluation, and interpretability using SHAP.

## Dataset

The dataset (`health_data.csv`) includes anonymized health information for multiple individuals. Features include:

- Age  
- BMI  
- Blood pressure  
- Glucose level  
- Smoking status  
- Alcohol consumption  
- Physical activity  
- Risk level (`Low`, `Medium`, `High`)

> Note: This dataset is synthetically generated for educational purposes.

## Technologies Used

- Python  
- Google Colab  
- Pandas, NumPy for data manipulation  
- Scikit-learn, XGBoost for machine learning  
- SHAP for model explainability  
- Matplotlib, Seaborn for visualization  
- Joblib for saving models

## Features

- Exploratory Data Analysis (EDA)  
- Preprocessing (handling missing values, encoding, scaling)  
- Training and comparing multiple models (Logistic Regression, Random Forest, XGBoost)  
- Hyperparameter tuning using GridSearchCV  
- Evaluation metrics: accuracy, precision, recall, F1-score, ROC AUC  
- Explainability using SHAP visualizations  
- Model saving and loading

## How to Run

### Option 1: Using Google Colab

1. Upload the following files to your Colab environment:
   - `health_risk_prediction_college_level.py`
   - `health_data.csv`

2. Install dependencies (if needed):

```bash
!pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap joblib
```

3. Run the script:

```bash
!python health_risk_prediction_college_level.py --data health_data.csv
```

Optional arguments:
- `--no-eda`: skip exploratory data analysis
- `--explain`: generate SHAP plots to explain model predictions

