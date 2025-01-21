![Screenshot 2025-01-21 055002](https://github.com/user-attachments/assets/62095a76-cd5d-40b1-a8ba-3cfbd1fadcd5)# Predict-Customer-Review-Score-For-Next-Purchase-MLOps 🌟

## Overview
This project implements a machine learning solution for predicting customer review scores using MLOps best practices. By leveraging ZenML and MLflow, we create a production-ready pipeline that predicts how likely a customer is to leave a positive review on their next purchase based on their historical behavior and transaction data.

## Problem Statement
We tackle the challenge of predicting customer review scores (typically on a 1-5 scale) based on various features including:
- Previous purchase history
- Customer demographics
- Product categories
- Order characteristics (delivery time, price, etc.)
- Historical review patterns
- Customer service interactions

Using this comprehensive dataset, we build and deploy a predictive model that helps businesses proactively identify potential customer satisfaction issues and take preventive actions.

## Purpose
This repository demonstrates the practical application of MLOps methodologies in building and deploying machine learning pipelines by:
- Providing a framework for customer satisfaction prediction
- Showcasing integration with MLflow for model tracking and monitoring
- Enabling automated retraining and deployment of models as new customer data becomes available

## 🐍 Installation

### Clone the Repository
```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-projects/customer-review-prediction
pip install -r requirements.txt
```

### Set Up ZenML Dashboard
ZenML 0.20.0+ includes a React-based dashboard for visualizing pipeline components:
```bash
pip install zenml["server"]
zenml up
```

### Install Required Integrations
For running the deployment pipeline:
```bash
zenml integration install mlflow -y
```

### Configure ZenML Stack
Set up a stack with MLflow components:
```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## 👍 Solution Architecture

### Training Pipeline
The training pipeline consists of five main steps:

1. **Data Ingestion**: 
   - Loads customer transaction history
   - Aggregates review data
   - Merges relevant customer information

2. **Data Preprocessing**:
   - Handles missing values
   - Encodes categorical variables
   - Creates feature aggregations (e.g., average past review scores)
   - Normalizes numerical features

3. **Feature Engineering**:
   - Calculates time-based features (days since last purchase)
   - Generates customer segment indicators
   - Creates interaction features between product categories and customer behavior

4. **Model Training**:
   - Trains the model with MLflow autologging
   - Implements cross-validation
   - Performs hyperparameter optimization

5. **Model Evaluation**:
   - Calculates key metrics (MAE, RMSE, R²)
   - Generates feature importance analysis
   - Performs bias testing across customer segments

### Deployment Pipeline
The deployment pipeline extends the training pipeline with:

1. **Deployment Trigger**:
   - Validates model performance meets minimum accuracy threshold
   - Checks for concept drift in predictions
   - Ensures fairness metrics are within acceptable ranges

2. **Model Deployer**:
   - Deploys the model as a REST API service using MLflow
   - Updates model artifacts and documentation
   - Maintains version control of deployed models

### Monitoring and Feedback Loop
- Tracks prediction accuracy over time
- Monitors feature drift and data quality
- Collects actual review scores for model retraining
- Triggers automated retraining when performance degrades

## 🚀 Usage

### Training Pipeline
To run the training pipeline:
```python
from pipelines.training_pipeline import train_pipeline
train_pipeline(data_path="data/customer_reviews.csv")
```

### Inference Pipeline
For making predictions:
```python
from utils.prediction import prediction_service_loader

service = prediction_service_loader(
    pipeline_name="continuous_deployment_pipeline",
    pipeline_step_name="mlflow_model_deployer_step",
    running=False,
)

# Example prediction
prediction = service.predict({
    'customer_id': '12345',
    'product_category': 'electronics',
    'purchase_amount': 299.99,
    'days_since_last_purchase': 30
})
```

## 📊 Dashboard
Launch the Streamlit dashboard for real-time predictions:
```bash
streamlit run streamlit_app.py
```




![Screenshot 2025-01-21 055002](https://github.com/user-attachments/assets/5f23ab0b-6273-4071-bee7-14b163bd17e1)



## 🔄 Production Deployment
For production environments, this solution can be extended with:
- Kubernetes deployment using Seldon
- Real-time feature serving using Feature Store
- A/B testing capabilities
- Automated retraining triggers
- Integration with customer feedback systems

## 📈 Metrics and Monitoring
The system tracks key metrics including:
- Prediction accuracy across different customer segments
- Model performance drift over time
- Feature importance stability
- Prediction latency and service reliability
- Customer segment coverage and bias metrics

## 🤝 Contributing
Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
