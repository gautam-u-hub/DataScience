from flask import Flask, request, jsonify
import pandas as pd
import pickle
from joblib import load

app = Flask(__name__)

# Load the CSV file containing default values
default_values_df = pd.read_csv('./newmode1.csv')

# Function to preprocess user inputs
def preprocess_user_inputs(user_inputs_df):
    # Drop unnecessary columns
    col_to_drop=['DOD', 'OperatingPhysician', 'OtherPhysician', 'ClmAdmitDiagnosisCode',
           'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
           'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
           'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
           'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
           'ClmProcedureCode_5', 'ClmProcedureCode_6','InscClaimAmtReimbursed']
    user_inputs_df.drop(columns=col_to_drop, axis=1, inplace=True)

    # Load the saved preprocessor
    loaded_preprocessor = load('healthcare_preprocessor.joblib')

    # Apply the preprocessor to the user inputs
    user_inputs_preprocessed = loaded_preprocessor.transform(user_inputs_df)
    return user_inputs_preprocessed

# Function to load the model
def load_model():
    # Load the pickled model
    with open('./RfPickelModelx1.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# @app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    input_data = pd.DataFrame([content])  # Wrap content in a list to create DataFrame with index
    
    # Preprocess user inputs
    user_inputs_preprocessed = preprocess_user_inputs(input_data)
    
    # Load the model
    model = load_model()
    
    # Make predictions
    predictions = model.predict(user_inputs_preprocessed)
    
    # Return predictions as JSON response
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=False)
