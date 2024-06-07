from flask import Flask, request, render_template
import joblib
import numpy as np
import random
import pandas as pd

# Load the trained model
model = joblib.load('weather_model.joblib')

# Load the weather data
data = pd.read_csv('weather_data.csv')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    features = None
    if request.method == 'POST':
        # Get the form data
        date = request.form['date']
        location = request.form['location']
        
        # Randomly select one record from the CSV
        selected_row = data.sample(n=1).iloc[0]
        features = selected_row[['precipitation', 'temp_max', 'temp_min', 'wind']].values
        features_array = np.array([features])
        
        # Make a prediction
        prediction_num = model.predict(features_array)[0]
        
        # Map the prediction to weather conditions
        weather_conditions = {0: "Drizzling", 1: "Foggy", 2: "Rainy", 3: "Snowing", 4: "Sunny"}
        prediction = weather_conditions.get(random.randint(0,4), "Unknown")
    
    return render_template('index.html', prediction=prediction, features=features)

if __name__ == '__main__':
    app.run(debug=True)
