from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.heartstrokeprediction.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)  # initializing a flask app

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Extract data from the form (matching your dataset columns)
            gender = int(request.form['gender'])
            age = float(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            ever_married = int(request.form['ever_married'])
            work_type = int(request.form['work_type'])
            Residence_type = int(request.form['Residence_type'])
            avg_glucose_level = float(request.form['avg_glucose_level'])
            bmi = float(request.form['bmi'])
            smoking_status = int(request.form['smoking_status'])
            
            # Prepare the input data (a list matching the model's expected input format)
            data = [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]
            data = np.array(data).reshape(1, -1)  # Ensure data is in 2D format (1 row, n columns)

            # Instantiate the PredictionPipeline class
            obj = PredictionPipeline()
            prediction = obj.predict(data)

            # Render results
            return render_template('results.html', prediction=str(prediction[0]))  # `prediction[0]` as it's a single value

        except Exception as e:
            print(f"The Exception message is: {e}")
            return 'There was an error processing your request.'

    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True, port=5000)  # running the flask app
