from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

def return_prediction(model, scaler, col_name, data):
    # Extract data from JSON
    Gender = data['gender']
    Marital_s = data['marital_status']
    No_Dependent = data['no_of_dependent']
    Type_residence = data['type_of_residence']
    Education = data['educational_attainment']
    Employment = data['employment_status']
    Sector = data['sector_of_employment']
    Amount = data['requested_amount']
    Purposeofloan = data['purpose']
    Age = data['age']
    Id = data['selfie_id_check']
    No_loans = data['loans']
    Telephone = data['phone_numbers']
    Os = data['mobile_os']
    Months_E = data['months_employed']
    Income = data['income_range']

    cat_df = pd.DataFrame([[Gender, Marital_s, Type_residence, Education, Employment, Sector, Purposeofloan, Id, Os, Income]],
                          columns=['gender', 'marital_status', 'type_of_residence', 'educational_attainment', 'employment_status',
                                   'sector_of_employment', 'purpose', 'selfie_id_check', 'mobile_os', 'income_range'])
    
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    num_df = pd.DataFrame([[No_Dependent, Amount, Age, No_loans, Months_E]],
                          columns=['no_of_dependent', 'requested_amount', 'age', 'loans', 'months_employed'])

    df = pd.concat([num_df, cat_encoded], axis=1)
    
    df = df.reindex(columns=col_name, fill_value=0)

    loan = scaler.transform(df.values)
    prediction = model.predict(loan)

    if prediction == 1:
        return 'Paid'
    else:
        return 'Not paid'



lendsqr = Flask(__name__)

@lendsqr.route("/")
def index():
    return '<h1>LENDSQR FLASK IS RUNNING</h1>'

lr_model = joblib.load('ldsqr_lr_model.pkl')
lr_scaler = joblib.load("ldsqr_scaler.pkl")
col_name = joblib.load("ldsqr_col_name.pkl")

@lendsqr.route('/loan_predict', methods=['POST'])
def loan_prediction():
    content = request.json
    result = return_prediction(lr_model, lr_scaler, col_name, content)
    return jsonify({'loan_status': result})

if __name__ == '__main__':
    lendsqr.run()
