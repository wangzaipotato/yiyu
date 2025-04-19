from model import allcause, card
from utils import fill_info, standarscaler, ai
import streamlit as st
import streamlit_shap
import pandas as pd
import shap
import joblib

st.title('Lifestyle-based mortality prediction tool')
st.divider()

with st.expander('click me to fill your information'):
    column1, column2, column3, column4 = st.columns([1,1,1,1])
    with column1:
        st.write('Demographic characteristic')
        age = st.number_input('Age', min_value=40, max_value=100, step=1)
        gender = st.selectbox(label='Gender',options=['Male', 'Female'])
        BMI = st.number_input('BMI', min_value=10.0, max_value=50.0, step=0.1)
        education = st.selectbox(label='Education level', options=['Less Than 9th Grade', '9-11th Grade', 'High School Grad/GED or Equivalent', 'Some College or AA degree', 'College Graduate or above'])
        ethnicity = st.selectbox( label='Ethnicity', options=['Mexican American', 'Other Hispanic', 'Non-Hispanic White', 'Non-Hispanic Black', 'Other Race - Including Multi-Racial'])
        INDFMPIR = st.number_input('The ratio of household income to poverty line', min_value=0.0, max_value=10.0, step=0.1)

    with column2:
        st.write('Medical history')
        diabetes = st.selectbox(label='Diabetes', options=['Diabetes', 'Prediabetes', 'No diabetes'])
        hypertension_level = st.selectbox(label='Hypertension', options=['Yes', 'No'])
        MCQ220 = st.selectbox('Have you ever been diagnosed with cancer?', options=['Yes', 'No'])
        MCQ160C = st.selectbox('Have you been diagnosed with coronary heart disease?', options=['Yes', 'No'])
        MCQ160F = st.selectbox('Have you been diagnosed with stroke?', options=['Yes', 'No'])


    with column3:
        st.write('Labortary test')
        non_HDL = st.number_input('non_HDL (mmol/L)')
        crp = st.number_input('CRP (mmol/L)')
        MCQ300A = st.selectbox('Close relative heart disease', options=['Yes', 'No'])
        MCQ300C = st.selectbox('Close relative diabetes', options=['Yes', 'No'])
        hbac1 = st.number_input('HbA1c (%)', min_value=0.0, max_value=100.0, step=1.0)

    with column4:
        st.write('Lifestyle')
        diet_score = st.number_input('Diet score', min_value=0.0, max_value=100.0, step=1.0)
        sedentary_minutes = st.number_input('Sedentary minutes', min_value=0, max_value=1000, step=10)
        physical_level = st.selectbox('Close relative heart disease', options=['Inactive', 'insufficient active', 'active', 'highly active'])
        sleep_level = st.selectbox('Sleep quality', ['Poor', 'Intermediate', 'Optimal'])
        drinks = st.number_input('Alcohol consumption per day (g)', min_value=0.0, max_value=100.0, step=1.0)
        smoke = st.selectbox('Smoking status', ['No smoking', 'Smoking before', 'Smoking'])  

prediction_button = st.button('Prediction')

info = pd.DataFrame(columns=['hypertension_level', 'BMXBMI', 'non_HDL', 'LBXCRP', 'RIDAGEYR',
       'RIAGENDR', 'RIDRETH1', 'DMDEDUC2', 'INDFMPIR', 'diabetes_level',
       'diet_score', 'MCQ300A', 'MCQ300C', 'physical_level',
       'Sedentary_minutes', 'sleep_level', 'drinks', 'smoking_status',
       'equal minutes', 'LBXGH', 'MCQ220', 'MCQ160C', 'MCQ160F'], data=None)

st.session_state['info'] = info
st.session_state['orinial_info'] = st.session_state['info'] = fill_info(info=info, hypertension_level=hypertension_level, BMI=BMI, non_HDL=non_HDL, crp=crp, age=age,
       gender=gender, ethnicity=ethnicity, education=education, INDFMPIR=INDFMPIR, diabetes=diabetes,
       diet_score=diet_score, MCQ300A=MCQ300A, MCQ300C=MCQ300C, physical_level=physical_level,
       sedentary_minutes=sedentary_minutes, sleep_level=sleep_level, drinks=drinks, smoke=smoke, hbac1=hbac1, MCQ220=MCQ220, MCQ160C=MCQ160C, MCQ160F=MCQ160F)

st.session_state['info'] = standarscaler(df=st.session_state['info'])
# 预测
if prediction_button:
    st.session_state['prediction_button'] = True
    st.write('All-cause mortality')
    st.session_state['allcause_mortality_risk'] = round(allcause(info=st.session_state['info']), 2)
    st.session_state['card_mortality_risk'] = round(card(info=st.session_state['info']), 2)
 
    st.write('Your 10 year all-cause mortality risk is {}'.format(round(st.session_state['allcause_mortality_risk'], 2)))

    allcause_explainer = shap.TreeExplainer(joblib.load('best_xgb_allcause.pkl'))
    allcause_shap_values = allcause_explainer.shap_values(st.session_state['info'])
    st.session_state['allcause_shap_values'] = allcause_shap_values

    shap_fig = shap.force_plot(
        allcause_explainer.expected_value, 
        allcause_shap_values[0,:], 
        st.session_state['info'].columns.tolist(),
        matplotlib=False,
        )
    streamlit_shap.st_shap(shap_fig)

    st.divider()

    st.write('Card mortality')
    card_mortality_risk = card(info=st.session_state['info'])
    card_mortality_risk = card(info=st.session_state['info'])

    st.write('Your 10 year cardicardiovascular mortality risk is {}'.format(round(st.session_state['card_mortality_risk'], 2)))

    card_explainer = shap.TreeExplainer(joblib.load('best_xgb_card.pkl'))
    card_shap_values = card_explainer.shap_values(st.session_state['info'])
    st.session_state['card_shap_values'] = card_shap_values
    shap_fig = shap.force_plot(
        card_explainer.expected_value, 
        card_shap_values[0,:], 
        st.session_state['info'].columns.tolist(),
        matplotlib=False,
        )
    streamlit_shap.st_shap(shap_fig)
    
ai_button = st.button('Get advice')
if ai_button:

    if not st.session_state['prediction_button']:
        st.info('Please click on Prediction button first!')
        st.stop()
    with st.spinner('Please wait a moment, suggestions are being generated'):
        response = ai(
            allcause_mortality_risk=st.session_state['allcause_mortality_risk'], 
            card_mortality_risk=st.session_state['card_mortality_risk'],
            allcause_shap=st.session_state['allcause_shap_values'],
            card_shap=st.session_state['card_shap_values']
        )
        st.info(response)
