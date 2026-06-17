import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv("cleaned_churn_data.csv")
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model to a file so we can reuse it without retraining
joblib.dump(model, 'churn_model.pkl')
print("Model saved as churn_model.pkl!")

# AGENT FUNCTION - automatically analyzes a customer
def analyze_customer(customer_data, customer_index):
    prediction = model.predict(customer_data)[0]
    probability = model.predict_proba(customer_data)[0][1]
    
    print(f"\n--- Customer {customer_index} Analysis ---")
    if prediction == 1:
        print(f"⚠️  ALERT: High churn risk! Probability: {probability:.2%}")
        print("Recommended Action: Offer discount or loyalty plan immediately")
    else:
        print(f"✅ Safe: Low churn risk. Probability: {probability:.2%}")

# Test the agent on first 5 customers from test set
print("\n=== AGENT RUNNING ON SAMPLE CUSTOMERS ===")
for i in range(5):
    sample = X_test.iloc[[i]]
    analyze_customer(sample, i)