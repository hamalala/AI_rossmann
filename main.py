import streamlit as st
import pickle
import numpy as np
import pandas as pd

# โหลด Model + Encoder
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoder'], data['cat_cols']

model, encoder, cat_cols = load_model()

# ฟีเจอร์ที่ใช้ตอนเทรน
features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'StateHoliday',
            'Month', 'Day', 'StoreType', 'Assortment', 'CompetitionDistance', 'Open', 'Customers']

# Default Values
default_values = {
    'Store': 1,
    'DayOfWeek': 2,
    'Promo': 1,
    'SchoolHoliday': 0,
    'StateHoliday': '0',
    'Month': 5,
    'Day': 10,
    'StoreType': 'c',
    'Assortment': 'a',
    'CompetitionDistance': 1200,
    'Open': 1,
    'Customers': 1000
}

st.title('🏪 Forecast Rossmann Store Sales')

st.header('Input Features')

# ====== Input Fields ======

# สร้างช่องกรอกตาม features
user_input = {}

# List ของ feature ที่เป็นตัวเลข
numeric_features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Month', 'Day', 'CompetitionDistance', 'Open', 'Customers']

for feature in features:
    if feature in numeric_features:
        user_input[feature] = st.number_input(
            f'{feature}', 
            value=float(default_values.get(feature, 0))
        )
    else:
        user_input[feature] = st.text_input(
            f'{feature}', 
            value=default_values.get(feature, '')
        )

# ====== กดปุ่ม Predict ======
if st.button('🚀 Forecast'):
    # ทำเป็น DataFrame
    input_df = pd.DataFrame([user_input])

    # แปลง categorical columns ด้วย encoder
    input_df[cat_cols] = input_df[cat_cols].astype(str)
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    # Predict
    prediction = model.predict(input_df)

    st.subheader('🔮 Predicted Sales:')
    st.success(f'{prediction[0]:,.2f}')
