from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("house_price_prediction.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            float(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            float(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT']),
        ]

        # Make prediction
        final_features = np.array([features])
        prediction = model.predict(final_features)

        output = round(prediction[0], 2)

        return render_template("index.html", prediction_text=f"Predicted Price: ${output}")

    except Exception as e:
        return render_template("index.html", prediction_text="Error in prediction: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
