#Streamlit app
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

#load the trained model
model = tf.keras.models.load_model('regression_model.h5')

#load the encoder and scaler
with open('label_encode.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('onehot_encode.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

## Streamlit app
st.title('Customer Churn of Estimated_Salary Prediction')

#User input
geography = st.selectbox('Geography',ohe.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])
Exited = st.selectbox('Exited',[0,1])

# proper the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [Exited]
})

# One - hot encode 'Geography'
geo_encoded = ohe.transform([input_data['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

# combine one-hot encode columns with input data
input_data = pd.concat([input_data.drop(['Geography'], axis=1), geo_encoded_df], axis=1)

# Encode categorical variable
input_data['Gender'] = label_encoder.transform(input_data['Gender'])

#Scaling the input data
input_scaled = scaler.transform(input_data)

#predict churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f'Estimated Salary: {prediction_proba:.2f}')


# if prediction_proba > 0.5:
#     st.write('The Customer is likely to churn')
# else:
#     st.write('The Customer is not likely to churn')


