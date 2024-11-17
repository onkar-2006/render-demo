from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Initialize the Flask app
app = Flask(__name__)

re = joblib.load('rf_model3_1.5.1.pkl')

def predict_diabetes_risk(features, model):
    """
    Predict the risk of diabetes and provide a risk score based on input features.
    """
    # Ensure features are in numpy array and reshaped correctly for model input
    features = np.array(features).reshape(1, -1)

    # Get prediction (binary class: 0 for Low Risk, 1 for High Risk)
    y_pred = model.predict(features)

    # Get the risk score (probability of the positive class - diabetes)
    risk_scores = model.predict_proba(features)[:, 1]  # Probability of positive class (1)

    # Convert prediction to readable format
    prediction = 'High risk' if y_pred[0] == 1 else 'Low risk'

    return prediction, risk_scores[0]

# Define route for HTML form
@app.route('/')
def home():
    return render_template('bp_pred.html')  # Serve the HTML file as the home page

# Define prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form submission
    data = request.form
    input_data = [
        int(data['Pregnancies']),
        int(data['Glucose']),
        int(data['BloodPressure']),
        int(data['SkinThickness']),
        int(data['Insulin']),
        int(data['BMI']),
        float(data['DiabetesPedigreeFunction']),
        int(data['Age'])
    ]

    # Run prediction
    prediction, risk_score = predict_diabetes_risk(input_data, re)

    # Return result as JSON
    return jsonify({"prediction": prediction, "risk_score": risk_score})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
