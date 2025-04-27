import streamlit as st
import pickle
import numpy as np

# โหลดโมเดล
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# ส่วนหัว
st.title('🚀 Forecast Rossmann Store Sales')

# Input Features
st.header('Input your features:')

# ====== รับ input ======
feature1 = st.number_input('Feature 1', value=0.0)
feature2 = st.number_input('Feature 2', value=0.0)
feature3 = st.number_input('Feature 3', value=0.0)

# ====== Predict ทันที ======
input_data = np.array([[feature1, feature2, feature3]])

prediction = model.predict(input_data)

# ====== แสดงผลทันที ======
st.subheader('🔮 Prediction Result:')
st.success(f'{prediction[0]:,.2f}')
