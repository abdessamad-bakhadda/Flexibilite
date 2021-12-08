# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

import EJP_Predictor_Deployment as epd


# Load the Random Forest CLassifier model
filename = 'EJP-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        jour = float(request.form['jour'])
        consommation = float(request.form['consommation'])
        temperature = float(request.form['temperature'])
        StockRestant = float(request.form['StockRestant'])

        print(jour,consommation, temperature, StockRestant)

        df_future = pd.DataFrame([[jour,consommation, temperature, StockRestant]],columns=["jour","consommation", "temperature", "StockRestant"])
        print(df_future)
        df_future = epd.preprocess(df_future, "future", epd.function_to_apply)
        print(df_future)
        my_prediction = classifier.predict(df_future)

        print(my_prediction ,type(my_prediction))
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)