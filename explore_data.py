import pandas as pd

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic info
print("Shape of data:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)

print("\nTotalCharges unique check:")
print(df['TotalCharges'].apply(type).value_counts())
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check how many became NaN (blank/invalid values)
print("\nAfter conversion - Missing in TotalCharges:")
print(df['TotalCharges'].isnull().sum())
# Remove rows with missing TotalCharges
df = df.dropna(subset=['TotalCharges'])

print("\nNew shape after removing missing values:")
print(df.shape)

# Also drop customerID - not useful for prediction
df = df.drop('customerID', axis=1)

print("\nFinal columns:")
print(df.columns.tolist())
# Convert Churn (target) to 0/1
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert all other categorical columns using One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

print("\nShape after encoding:")
print(df_encoded.shape)

print("\nSample columns after encoding:")
print(df_encoded.columns.tolist()[:10])

# Save cleaned data for next step
df_encoded.to_csv("cleaned_churn_data.csv", index=False)
print("\nCleaned data saved successfully!")