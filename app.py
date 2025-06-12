import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory

# Load the trained model
model = joblib.load('titanic_rf_model.pkl')

app = Flask(__name__, template_folder='.')
# @app.route('/style.css')
# def serve_css():
#     return send_from_directory('.', 'style.css', mimetype='text/css')


@app.route('/')
def home():
    return render_template('index.html', prediction=None)  # Render the web page with no prediction initially

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'Pclass': int(request.form['Pclass']),
            'Sex': request.form['Sex'],
            'Age': float(request.form['Age']),
            'SibSp': int(request.form['SibSp']),
            'Parch': int(request.form['Parch']),
            'Fare': float(request.form['Fare']),
            'Embarked': request.form['Embarked']
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Convert categorical data to numerical (Assuming model was trained with encoded values)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Adjust based on model training
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Adjust if different encoding was used
        
        # Ensure the feature order matches training data
        expected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        df = df.reindex(columns=expected_features)

        # Make prediction
        prediction = model.predict(df)[0]
        prediction_text = "Survived" if prediction == 1 else "Did Not Survive"

        # Return the result to the webpage
        return render_template('index.html', prediction=prediction_text)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='192.168.0.105')

