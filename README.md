# Customer Churn Prediction ML Pipeline

## 📋 Project Overview
An end-to-end machine learning pipeline for predicting telecom customer churn using Scikit-learn. This production-ready solution demonstrates comprehensive ML workflow from data preprocessing to model deployment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

## 🎯 Objective
Build a reusable, production-ready ML pipeline for customer churn prediction that:
- Handles mixed data types (numerical & categorical)
- Implements proper preprocessing and feature engineering
- Trains and compares multiple ML models
- Performs hyperparameter tuning with GridSearchCV
- Exports complete pipeline for deployment

## 📊 Dataset
**Telco Customer Churn Dataset** from IBM:
- 7,043 customer records with 21 features
- **Target**: `Churn` (Yes/No) - 26.5% churn rate
- **Key Features**: tenure, monthly charges, contract type, payment method, internet service

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                 End-to-End ML Pipeline                   │
├─────────────────────────────────────────────────────────┤
│ 1. Data Loading & Validation                            │
│ 2. Exploratory Data Analysis (EDA)                      │
│ 3. Train-Test Split (Stratified)                        │
│ 4. Preprocessing Pipeline                               │
│    ├─ Numeric: Imputation + Scaling                     │
│    └─ Categorical: Imputation + One-Hot Encoding        │
│ 5. Model Pipeline Construction                          │
│ 6. Hyperparameter Tuning (GridSearchCV)                 │
│ 7. Model Evaluation & Comparison                        │
│ 8. Pipeline Serialization (Joblib)                      │
│ 9. Production Inference Class                           │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Features

### ✅ Core Features
- **Data Validation**: Automatic data quality checks and cleaning
- **Feature Engineering**: Domain-specific feature creation
- **Pipeline Construction**: Complete Scikit-learn Pipeline API implementation
- **Model Training**: Logistic Regression, Random Forest, Gradient Boosting
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Model Export**: Complete pipeline serialization with metadata

### ✅ Production-Ready Features
- **Modular Design**: Separate classes for validation, pipeline, inference
- **Error Handling**: Robust input validation and error management
- **Logging**: Comprehensive execution logging
- **Configuration**: Configurable pipeline parameters
- **Reusability**: Ready for API deployment or batch processing

## 📁 Project Structure
```
customer-churn-pipeline/
├── telco_churn_pipeline.ipynb          # Main Jupyter notebook
├── telco_churn_pipeline.joblib         # Complete pipeline (with metadata)
├── telco_churn_model.pkl              # Lightweight model
├── pipeline_execution.log             # Execution logs
├── requirements.txt                   # Dependencies
└── README.md                         # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/customer-churn-pipeline.git
cd customer-churn-pipeline
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Jupyter Notebook
```bash
jupyter notebook telco_churn_pipeline.ipynb
```

## 📖 Usage

### Option 1: Complete Pipeline Execution
Run the entire notebook cell-by-cell to:
1. Load and explore data
2. Train models with hyperparameter tuning
3. Evaluate performance
4. Export pipeline

### Option 2: Production Inference
```python
from inference import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor('telco_churn_pipeline.joblib')

# Make prediction for a customer
customer_data = {
    'tenure': 12,
    'MonthlyCharges': 80.5,
    'TotalCharges': 966.0,
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Contract': 'Month-to-month',
    'PaymentMethod': 'Electronic check'
}

result = predictor.predict(customer_data)
print(result)
```

### Option 3: Batch Prediction
```python
# Predict for multiple customers
customers_list = [...]
batch_results = predictor.predict_batch(customers_list)
```

## 📈 Results

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8013 | 0.6486 | 0.5401 | 0.5894 | 0.8385 |
| **Random Forest (Best)** | **0.7991** | **0.6810** | 0.4733 | 0.5583 | **0.8506** |

### Key Insights
1. **Best Model**: Random Forest with balanced class weights
2. **ROC AUC**: 0.8506 (excellent discriminative power)
3. **Top Features**: Tenure, Monthly Charges, Contract Type
4. **Business Impact**: Can identify 54% of churning customers with 65% precision

## 🛠️ Technical Implementation

### Preprocessing Pipeline
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)
```

### Complete ML Pipeline
```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

### Hyperparameter Tuning
```python
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)
```

## 🔄 Deployment Options

### Option A: REST API (FastAPI)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
predictor = ChurnPredictor('telco_churn_pipeline.joblib')

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    # ... other features

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    return predictor.predict(customer.dict())
```

### Option B: Batch Processing
```python
import pandas as pd

def batch_predict(csv_file):
    df = pd.read_csv(csv_file)
    predictions = []
    for _, row in df.iterrows():
        pred = predictor.predict(row.to_dict())
        predictions.append(pred)
    return pd.DataFrame(predictions)
```

### Option C: Cloud Deployment (AWS SageMaker)
1. Package model as Docker container
2. Deploy to SageMaker endpoint
3. Set up auto-scaling and monitoring

## 📊 Monitoring & Maintenance

### Model Monitoring
- **Performance Drift**: Track ROC AUC degradation
- **Data Drift**: Monitor feature distribution changes
- **Concept Drift**: Watch for changes in churn patterns

### Retraining Strategy
- **Trigger**: Monthly or when performance drops below threshold
- **Data**: Use rolling window of 6 months
- **Validation**: A/B testing with current model

## 🧪 Testing

### Unit Tests
```python
def test_data_validation():
    validator = DataValidator()
    df = validator.validate_and_load_data('sample.csv')
    assert 'Churn' in df.columns

def test_pipeline_prediction():
    predictor = ChurnPredictor('model.joblib')
    result = predictor.predict(sample_customer)
    assert 'churn_prediction' in result
```

### Integration Tests
- End-to-end pipeline execution
- Model loading and inference
- Error handling scenarios

## 📝 Code Quality
- **Type Hints**: Comprehensive type annotations
- **Docstrings**: Detailed function documentation
- **Logging**: Structured logging for debugging
- **Error Handling**: Graceful degradation and user feedback

## 🚨 Error Handling
The pipeline includes robust error handling for:
- Missing data files
- Invalid input data
- Model loading failures
- Prediction errors

## 📚 Dependencies

### Core Dependencies
```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

### Optional Dependencies (for API deployment)
```txt
fastapi>=0.95.0
uvicorn>=0.21.0
pydantic>=1.10.0
```

## 🏆 Performance Metrics

### Model Metrics
- **ROC AUC**: 0.8506
- **Precision**: 0.6810
- **Recall**: 0.4733
- **F1 Score**: 0.5583

### Pipeline Metrics
- **Training Time**: 8.5 minutes (full GridSearchCV)
- **Prediction Time**: < 10ms per customer
- **Model Size**: 4.76 MB (with metadata)

## 🔮 Future Enhancements

### Planned Features
1. **Real-time Inference**: WebSocket support for streaming predictions
2. **Dashboard**: Interactive model performance dashboard
3. **A/B Testing**: Framework for model comparison
4. **Automated Retraining**: CI/CD pipeline for model updates
5. **Explainability**: SHAP/LIME integration for predictions

### Research Directions
- Experiment with deep learning models
- Implement ensemble methods
- Add time-series features for tenure analysis
- Incorporate external data sources

## 👥 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

### Code Standards
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings for all functions
- Include unit tests for new features


## 🙏 Acknowledgments

### Dataset
- **Source**: IBM Telco Customer Churn Dataset
- **Provider**: IBM Developer

### Tools & Libraries
- Scikit-learn for ML pipeline framework
- Pandas for data manipulation
- Joblib for model serialization

### Inspiration
- Scikit-learn documentation and examples
- Machine learning engineering best practices
- Production ML system design patterns

## 👤 Author
**Youmna Saifullah**  
*ML Engineer Intern*  

## 📞 Contact
For questions, feedback, or collaboration opportunities:
- **Email**: youmna.saifullah@gmail.com


## 🌟 Show Your Support
If you find this project useful, please give it a ⭐️ on GitHub!

---
