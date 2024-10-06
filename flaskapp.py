from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd



def validate_input(data):
    required_fields = ['gender', 'type_of_residence', 'educational_attainment', 'employment_status',
                       'sector_of_employment', 'requested_amount', 'purpose', 'loan_request_day',
                       'age', 'selfie_id_check', 'loans', 'phone_numbers', 'mobile_os', 'income_range']

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing fields: {', '.join(missing_fields)}"

    # Ensure numeric fields are actually numbers
    numeric_fields = ['requested_amount', 'age', 'loans', 'phone_numbers']
    for field in numeric_fields:
        if not isinstance(data[field], (int, float)):
            return False, f"Invalid type for {field}, expected a number"
    
    # Range checks
    if not (100 <= data['requested_amount'] <= 2500000):
        return False, "Invalid 'requested_amount', must be between 100 and 2,500,000"

    if not (18 <= data['age'] <= 67):
        return False, "Invalid 'age', must be between 18 and 67"

    if not (0 <= data['phone_numbers'] <= 5):
        return False, "Invalid 'phone_numbers', must be between 1 and 5"

    return True, "Valid input"



def return_prediction(model, scaler, col_name, data):
    Gender = data['gender']
    Type_residence = data['type_of_residence']
    Education = data['educational_attainment']
    Employment = data['employment_status']
    Sector = data['sector_of_employment']
    Amount = data['requested_amount']
    Purposeofloan = data['purpose']
    Loan_day = data['loan_request_day']
    Age = data['age']
    Id = data['selfie_id_check']
    No_loans = data['loans']
    Telephone = data['phone_numbers']
    Os = data['mobile_os']
    Income = data['income_range']

    cat_df = pd.DataFrame([[Gender, Type_residence, Education, Employment, Sector, Purposeofloan, Loan_day, Id, Os, Income]],
                          columns=['gender', 'type_of_residence', 'educational_attainment', 'employment_status',
                                   'sector_of_employment', 'purpose', 'loan_request_day', 'selfie_id_check', 'mobile_os', 'income_range'])
    
    cat_encoded = pd.get_dummies(cat_df, drop_first=True)

    num_df = pd.DataFrame([[Amount, Age, No_loans, Telephone]],
                          columns=['no_of_dependent', 'age', 'loans', 'phone_numbers'])

    df = pd.concat([num_df, cat_encoded], axis=1)
    
    df = df.reindex(columns=col_name, fill_value=0)

    loan = scaler.transform(df.values)
    prediction = model.predict(loan)

    if prediction == 1:
        return 'Paid'
    else:
        return 'Not paid'

cat_feat = ['gender','employment_status','sector_of_employment','loan_request_day','selfie_id_check','type_of_residence', 'educational_attainment','purpose','mobile_os','income_range']

num_feat = ['age','requested_amount','loans','phone_numbers']

lendsqr = Flask(__name__)

@lendsqr.route("/")
def index():
    return '<h1>LENDSQR LOAN APP IS RUNNING</h1>'

lr_model = joblib.load('ldsqr_lr_model.pkl')
lr_scaler = joblib.load("ldsqr_scaler.pkl")
col_name = joblib.load("col_name.pkl")

@lendsqr.route('/loan_predict', methods=['POST'])
def loan_prediction():
    content = request.json
    is_valid, message = validate_input(content)
    if not is_valid:
        return jsonify({'error': message}), 400
    result = return_prediction(lr_model, lr_scaler, col_name, content)
    return jsonify({'loan_status': result})

if __name__ == '__main__':
    lendsqr.run()
