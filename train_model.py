import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_data

X_scaled, y, le, scaler = preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
joblib.dump(le, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"✅ Model trained and saved. Test accuracy: {model.score(X_test, y_test):.4f}")
