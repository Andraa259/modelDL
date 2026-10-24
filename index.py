import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

feature_names = [
    'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
    'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic',
    'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma',
    'KidneyDisease', 'SkinCancer'
]

st.title('Prediksi Risiko Penyakit Jantung')

bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
smoking = st.selectbox('Smoking', ['Yes', 'No'])
alcohol = st.selectbox('Alcohol Drinking', ['Yes', 'No'])
stroke = st.selectbox('Stroke', ['Yes', 'No'])
physical_health = st.number_input('Physical Health (hari tidak sehat)', min_value=0, max_value=30, value=0)
mental_health = st.number_input('Mental Health (hari tidak sehat)', min_value=0, max_value=30, value=0)
diff_walking = st.selectbox('Difficulty Walking', ['Yes', 'No'])
sex = st.selectbox('Sex', ['Male', 'Female'])
age_category = st.selectbox('Age Category', ['18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80 or older'])
diabetic = st.selectbox('Diabetic', ['Yes','No','No, borderline diabetes','Yes (during pregnancy)'])
physical_activity = st.selectbox('Physical Activity', ['Yes', 'No'])
gen_health = st.selectbox('General Health', ['Excellent','Very good','Good','Fair','Poor'])
sleep_time = st.number_input('Sleep Time (jam)', min_value=0.0, max_value=24.0, value=7.0)
asthma = st.selectbox('Asthma', ['Yes', 'No'])
kidney_disease = st.selectbox('Kidney Disease', ['Yes', 'No'])
skin_cancer = st.selectbox('Skin Cancer', ['Yes', 'No'])

if st.button('Prediksi'):
    data = {
        'BMI': bmi,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol,
        'Stroke': stroke,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'DiffWalking': diff_walking,
        'Sex': sex,
        'AgeCategory': age_category,
        'Diabetic': diabetic,
        'PhysicalActivity': physical_activity,
        'GenHealth': gen_health,
        'SleepTime': sleep_time,
        'Asthma': asthma,
        'KidneyDisease': kidney_disease,
        'SkinCancer': skin_cancer
    }
    
    input_data = []
    for feature in feature_names:
        value = data[feature]
        if feature in label_encoders:
            value = label_encoders[feature].transform([value])[0]
        else:
            value = float(value)
        input_data.append(value)
    
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_reshaped = input_scaled.reshape(1, len(feature_names), 1)
    
    prob = model.predict(input_reshaped)[0][0]
    result = "Yes" if prob > 0.5 else "No"
    
    st.write(f"**Risiko Penyakit Jantung:** {result}")
    st.write(f"**Probabilitas:** {prob * 100:.2f}%")
