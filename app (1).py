
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Customer Churn Prediction App')
st.markdown('Enter customer details to predict the likelihood of churning.')

# User input fields in a sidebar for a clean UI
st.sidebar.header('Customer Information')

credit_score = st.sidebar.slider('Credit Score', 300, 850, 650)
age = st.sidebar.slider('Age', 18, 100, 40)
tenure = st.sidebar.slider('Tenure (Years with bank)', 0, 10, 5)
balance = st.sidebar.number_input('Account Balance', value=75000.0)
products_number = st.sidebar.slider('Number of Products', 1, 4, 1)
credit_card = st.sidebar.selectbox('Has a Credit Card?', ['Yes', 'No'])
active_member = st.sidebar.selectbox('Is an Active Member?', ['Yes', 'No'])
estimated_salary = st.sidebar.number_input('Estimated Salary', value=100000.0)
country = st.sidebar.selectbox('Country', ['France', 'Germany', 'Spain'])
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])

# Convert user input into a DataFrame
def get_user_data():
    data = {
        'credit_score': credit_score,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': 1 if credit_card == 'Yes' else 0,
        'active_member': 1 if active_member == 'Yes' else 0,
        'estimated_salary': estimated_salary,
        'country_Germany': 1 if country == 'Germany' else 0,
        'country_Spain': 1 if country == 'Spain' else 0,
        'gender_Male': 1 if gender == 'Male' else 0
    }
    df = pd.DataFrame(data, index=[0])
    return df

user_df = get_user_data()

# Scale numerical features
numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])

# Make prediction
prediction = model.predict(user_df)
prediction_proba = model.predict_proba(user_df)

st.write('---')
st.header('Prediction Result')
if prediction[0] == 1:
    st.error('This customer is likely to churn.')
else:
    st.success('This customer is unlikely to churn.')
st.write(f"Confidence: {prediction_proba[0][int(prediction[0])]:.2f}")

st.markdown('---')
st.markdown("### How to Deploy this App")
st.markdown("1. Download all four files from the left pane: `churn_model.pkl`, `scaler.pkl`, `app.py`, and `requirements.txt`.")
st.markdown("2. Create a **public** GitHub repository and upload all four files to it.")
st.markdown("3. Go to your Streamlit Cloud dashboard, select 'New app', and choose your repository to deploy.")
