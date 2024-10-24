from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

# Load the model and encoders
model = joblib.load('flight_model.pkl')
encoders = joblib.load("encoders.pkl")
df = pd.read_csv("flight_data.csv")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get the values from the form
        flight_num = request.form['flight']
        carrier = request.form['carrier']
        origin = request.form['origin']
        dest = request.form['dest']
        distance = int(request.form['distance'])
        hour = int(request.form['hour'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        schedule_hr=int(request.form['shour'])
        arrival_hr=int(request.form['ahour'])

        # Prepare the input data
        input_data = pd.DataFrame({         
            'flight': [flight_num],  
            'carrier': [carrier],
            'origin': [origin],
            'dest': [dest],
            'distance': [distance],
            'hour': [hour],
            'day': [day],
            'month': [month],
            'schedule_hr':[schedule_hr],
            'arrival_hr':[arrival_hr]
        })
        
        # Apply encoders on top of the input data
        input_data=input_data.iloc[:,1:8]
        for col in encoders.keys():
            input_data[col] = encoders[col].transform(input_data[col])[0]

        # Make the prediction
        prediction = model.predict(input_data.values)
        result = 'This flight is likely to be departing late. Thank You for your Cooperation.' if prediction[0] == 1 else 'This flight is likely to be departing on time.'
        
        return render_template('index.html', result=result, carrier=carrier, origin=origin, dest=dest, distance=distance, hour=hour, day=day, month=month,
                               carriers=df['carrier'].unique(), origins=df['origin'].unique(), destinations=df['dest'].unique())

    return render_template('index.html', result=None, carriers=df['carrier'].unique(), origins=df['origin'].unique(), destinations=df['dest'].unique())

if __name__ == "__main__":
    app.run(debug=True)
