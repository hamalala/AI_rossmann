import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import date

# โหลด Model + Encoder
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoder'], data['cat_cols']

model, encoder, cat_cols = load_model()

features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'StateHoliday',
            'Month', 'Day', 'StoreType', 'Assortment', 'CompetitionDistance', 'Open', 'Customers']

st.title('🏪 Forecast Rossmann Store Sales')
st.header('Input Features')

user_input = {}

# ====== กรอกข้อมูล ======
user_input['Store'] = st.number_input('Store (ID)', min_value=1, step=1, value=1)

selected_date = st.date_input('Select Date', value=date(2025, 5, 1))
user_input['DayOfWeek'] = selected_date.weekday() + 1
user_input['Month'] = selected_date.month
user_input['Day'] = selected_date.day

user_input['Promo'] = st.radio('Promo', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
user_input['SchoolHoliday'] = st.radio('School Holiday', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
user_input['StateHoliday'] = st.radio('State Holiday', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
user_input['StoreType'] = st.radio('Store Type', options=['a', 'b', 'c'])
user_input['Assortment'] = st.radio('Assortment Type', options=['a', 'b', 'c'])

user_input['CompetitionDistance'] = st.number_input('Competition Distance', min_value=0.0, step=100.0, value=1200.0)
user_input['Open'] = st.radio('Store Open?', options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
user_input['Customers'] = st.number_input('Expected Customers', min_value=0.0, step=100.0, value=1000.0)

# ====== กดปุ่ม Predict ======
if st.button('🚀 Forecast'):
    input_df = pd.DataFrame([user_input])
    input_df[cat_cols] = input_df[cat_cols].astype(str)
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    prediction = model.predict(input_df)
    st.subheader('🔮 Predicted Sales:')
    st.success(f'{prediction[0]:,.2f}')
