from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/predict_start_date', methods=['POST'])
def predict_start_date():
    try:
        # Receive data from the request body
        data = request.get_json()
        # Prepare the data in a DataFrame format
        df = pd.DataFrame({
            'date': [datetime(entry["startYear"], entry["startMonth"], entry["startDay"]) for entry in data],
            'afterDays': [entry["afterDays"] for entry in data]
        })

        # Train the ARIMA model
        model = ARIMA(df['afterDays'], order=(1, 1, 1))  # Adjust the order as needed
        fit_model = model.fit()

        # Forecast the "afterDays" for the next cycle
        forecast = fit_model.get_forecast(steps=1)
        predicted_days = forecast.predicted_mean.values[0]

        # Calculating the predicted start date
        last_entry = data[-1]
        last_cycle_end_date = datetime(last_entry["endYear"], last_entry["endMonth"], last_entry["endDay"])
        predicted_start_date = last_cycle_end_date + timedelta(days=int(predicted_days))

        # Return the predicted start date as JSON
        response = {
            "predicted_start_date": predicted_start_date.strftime("%d %B %Y")
        }
        print(response)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
