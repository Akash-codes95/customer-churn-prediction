customer-churn-prediction
Agentic AI system for customer churn prediction

Customer Churn Prediction - Agentic AI System

An end-to-end Machine Learning system that predicts customer churn risk and automatically recommends retention actions, built with an agentic AI approach.

# Project Overview

This project analyzes customer data to predict which customers are likely to stop using a service (churn). Instead of just outputting a number, the system acts like an automated agent — it analyzes each customer, flags high-risk cases, and suggests a recommended action, with zero manual intervention needed.

# Problem Statement

Customer retention is far cheaper than acquiring new customers. Businesses need a way to identify at-risk customers *before* they leave, so they can act proactively rather than reactively.

# Tech Stack

- # Python - Core programming language
- # Pandas & NumPy - Data cleaning and manipulation
- # Scikit-learn - Machine Learning (Random Forest Classifier)
- # Streamlit - Interactive web dashboard
- # Joblib - Model serialization

# Dataset

[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - 7,043 customer records with 21 features including demographics, account info, and services used.

# Project Workflow

1. **Data Cleaning** (`explore_data.py`) - Handled missing values, converted data types, encoded categorical variables
2. **Model Training** (`train_model.py`) - Trained a Random Forest Classifier, achieving 78.5% accuracy
3. **Agentic Prediction System** (`agent.py`) - Automated agent that analyzes customers and recommends actions based on risk level
4. **Interactive Dashboard** (`dashboard.py`) - Live Streamlit dashboard showing predictions and business metrics

# Results

- **Model Accuracy:** 78.5%
- **Overall Churn Rate Identified:** 26.6%
- Successfully flags high-risk customers with actionable recommendations in real time

# How to Run

```bash
pip install pandas numpy scikit-learn streamlit joblib
python explore_data.py
python agent.py
streamlit run dashboard.py
```

# Future Improvements

- Improve recall for churn class using SMOTE/class balancing
- Add LLM-based natural language explanations for each prediction
- Deploy to cloud (Streamlit Cloud / AWS)
- Add automated email alerts for high-risk customers

# Author

**Akash Maddineni**  
MSc Artificial Intelligence Student, ULSTER UNIVERSITY, Northren ireland.
