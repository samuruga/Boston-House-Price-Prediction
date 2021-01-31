#################################################################################################
# Project: Take home test
# Version: 1.0
# Date: 1/31/2021
# Description: Flask based API server to serve ML model
# Author: Saranya Murugan
#################################################################################################

# Imported modules below
from flask import Flask, request, Response
import joblib
import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import statistics as stat
import pickle
from flasgger import Swagger

# Global variables below
app = Flask(__name__)

# Create the swagger object
swagger = Swagger(app)

# Loading model file
model_file = "models\\final_model.pkl"

# Loading scaler objects for MinMaxScaler function
X_scaler = "models\\x_scaler.pkl"
Y_scaler = "models\\y_scaler.pkl"

# Functions below

# Function Name: data_clean
# Input arguments: data (Type: pandas dataframe) - Input is unclean data
# Output values: data (Type: pandas dataframe) - Output is cleaned data
def data_clean(data):
    data['CHAS'].fillna(stat.mode(data['CHAS']), inplace = True)
    data['CRIM'].fillna(data['CRIM'].median(),inplace = True)
    data['ZN'].fillna(data['ZN'].median(), inplace = True)
    data['B'].fillna(data['B'].median(), inplace = True)
    data['LSTAT'].fillna(data['LSTAT'].mean(),inplace = True)
    data['AGE'].fillna(data['AGE'].mean(), inplace = True)
    data['INDUS'].fillna(data['INDUS'].mean(), inplace = True)
    return data

# API routes definitions below

# Home page
# Methods: GET
@app.route('/')
def home():
    return 'To try inference model, navigate to "/apidocs" end point'

# Predict API
# Methods: POST    
@app.route('/predict', methods=["POST"])
def predict():
    """Predict API to predict house prices in Boston
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Succeeded
      400:
        description: Uploaded file should be of type csv
    """
    try:    
        # Unpickle the pickled model file and x and y scalers from training set
        logistic_from_joblib = pickle.load(open(model_file, 'rb'))
        x_scaler = pickle.load(open(X_scaler, 'rb'))
        y_scaler = pickle.load(open(Y_scaler, 'rb'))
        
        # Load the test data
        test_data = pd.read_csv(request.files.get("file"))
        
        # Clean the test data
        clean_test_data = data_clean(test_data)
        
        # Pre-process the test data
        cols = ['CRIM','ZN','INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO','LSTAT']
        x = clean_test_data.loc[:,cols]
        for col in x.columns:
            if np.abs(x[col].skew()) > 0.3:
                x[col] = np.log1p(x[col])
        test_data_scaled = x_scaler.transform(x)
        
        # Predict
        y_pred = logistic_from_joblib.predict(test_data_scaled)
        
        # Transform results back to original scales
        Y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1))
        
        # Create the output dataframe
        final_pred = pd.DataFrame(Y_pred)
        #prediction = model.predict(np.array(input_data))
        final_pred.columns =['Predicted House Prices(k$)']
        final_data = pd.concat([test_data, pd.DataFrame(final_pred)], axis=1)
        
        return (str(final_data))
    
    except Exception as error:
        return Response(str("Uploaded file should be of type csv"), status=400)

    
if __name__ == '__main__':
    app.run()

# End of code
#################################################################################################