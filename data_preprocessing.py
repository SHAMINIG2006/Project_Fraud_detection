import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data():
    df = pd.read_csv("dataset.csv", low_memory=False)
    df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

    # Encode transaction type
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    # Ensure all feature columns are numeric
    feature_cols = ['step', 'type', 'amount', 'oldbalanceOrg',
                    'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df.dropna(inplace=True)

    X = df[feature_cols]
    y = df['isFraud']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, scaler
