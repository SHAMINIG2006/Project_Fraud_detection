from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Gather all 8 features in correct order
    fv = [
        float(request.form['step']),
        encoder.transform([request.form['type']])[0],
        float(request.form['amount']),
        float(request.form['oldbalanceOrg']),
        float(request.form['newbalanceOrig']),
        float(request.form['oldbalanceDest']),
        float(request.form['newbalanceDest']),
        int(request.form['isFlaggedFraud'])
    ]

    X_input = np.array([fv])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    result = "Fraudulent" if pred == 1 else "Genuine"

    return render_template('result.html', prediction=result)

@app.route('/exit')
def exit_view():
    return render_template('exit.html')

if __name__ == '__main__':
    app.run(debug=True)
