import streamlit as st
import pandas as pd
import joblib

# Page title
st.title("🔮 Customer Churn Prediction Dashboard")
st.write("Agentic AI system to predict customer churn risk")

# Load the trained model
model = joblib.load('churn_model.pkl')

# Load the cleaned data to get column names
df = pd.read_csv("cleaned_churn_data.csv")
feature_columns = df.drop('Churn', axis=1).columns

st.header("📊 Sample Predictions")

# Show first 10 customers from data with predictions
sample_data = df.drop('Churn', axis=1).head(10)
predictions = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)[:, 1]

# Build a results table
results = sample_data.copy()
results['Churn Risk %'] = (probabilities * 100).round(2)
results['Prediction'] = ['⚠️ High Risk' if p == 1 else '✅ Low Risk' for p in predictions]

st.dataframe(results[['Churn Risk %', 'Prediction']])

st.header("📈 Overall Stats")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Customers Analyzed", len(df))
with col2:
    churn_rate = (df['Churn'].sum() / len(df)) * 100
    st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")