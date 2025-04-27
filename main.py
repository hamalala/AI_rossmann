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

st.title('🏪 Forecast Rossmann Store Sales')

st.header('Input Features')

# ====== Input Fields ======

# สร้างช่องกรอกตาม features
user_input = {}

# List ของ feature ที่เป็นตัวเลข (กรอกเป็น number_input)
numeric_features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Month', 'Day', 'CompetitionDistance', 'Open', 'Customers']

# สร้างช่อง input
for feature in features:
    if feature in numeric_features:
        user_input[feature] = st.number_input(f'{feature}', value=0.0)
    else:
        user_input[feature] = st.text_input(f'{feature}', value='')

# ====== ทำเป็น DataFrame ======
input_df = pd.DataFrame([user_input])

# แปลง categorical columns ด้วย encoder
input_df[cat_cols] = input_df[cat_cols].astype(str)
input_df[cat_cols] = encoder.transform(input_df[cat_cols])

# ====== Predict ทันที ======
prediction = model.predict(input_df)

st.subheader('🔮 Predicted Sales:')
st.success(f'{prediction[0]:,.2f}')
