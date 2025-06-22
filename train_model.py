import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data
import pandas as pd

# Load data and reduce size
df = pd.read_csv("dataset.csv", low_memory=False).sample(n=50000, random_state=42)
df.to_csv("sampled_dataset.csv", index=False)  # Save to reuse

# Modify preprocess_data to read sampled file
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_sampled():
    df = pd.read_csv("sampled_dataset.csv")
    df.drop(['nameOrig', 'nameDest'], axis=1, errors='ignore', inplace=True)

    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    feature_cols = ['step', 'type', 'amount', 'oldbalanceOrg',
                    'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    X = df[feature_cols]
    y = df['isFraud']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, scaler

# Use the optimized preprocessing
X_scaled, y, le, scaler = preprocess_sampled()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train faster with fewer trees and parallel CPU use
model = RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Save all required files
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"✅ Model trained on 50,000 rows and saved. Accuracy: {model.score(X_test, y_test):.4f}")
