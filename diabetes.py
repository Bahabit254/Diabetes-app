import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import joblib
import sklearn


def main():
    st.title('Diabetes prediction using Extra Trees')
    
    filename='extc.pkl'
    loaded_model=joblib.load(filename)
    
    
    
    col_1, col_2=st.columns(2)
    with col_1:
        Age=st.number_input('Age',min_value=1,max_value=100)

        Pregnancies=st.number_input('Pregnacy',min_value=0,max_value=20)

        DiabetesPedigreeFunction=st.number_input('Pedigree_function',min_value=0.00,max_value=5.00)

        BMI=st.number_input('Body_mass_index',min_value=0.00,max_value=100.00)

    with col_2:
        Glucose=st.number_input('Glucose_level',min_value=0.00,max_value=300.0)

        BloodPressure=st.number_input('Blood pressure',min_value=50,max_value=300)

        SkinThickness=st.number_input('Triceps skinfold thickness',min_value=0.00,max_value=150.00)

        Insulin=st.number_input('Insulin levels',min_value=0.00,max_value=950.00)



    input_dict={'Age':Age,'Pregnancies':Pregnancies,'DiabetesPedigreeFunction':DiabetesPedigreeFunction,'BMI':BMI,'Glucose':Glucose,'BloodPressure':BloodPressure,'SkinThickness':SkinThickness,'Insulin':Insulin}
    input_df=pd.DataFrame(input_dict,index=[0])

    Button=st.button('Predict')

    if Button:
        Diabetes= loaded_model.predict(input_df)
        if Diabetes==0:
            st.success('Low risk of diabetes')
        else:
            st.error('High risk of diabetes')


        precision, recall, f1, acc = st.columns(4)
        st.markdown("""<style>div[data-testid="metric-container"] {background-color: rgba(28, 131, 225, 0.1);border: 1px solid rgba(28, 131, 225, 0.1);padding: 5% 5% 5% 10%;border-radius: 5px;color: rgb(30, 103, 119);overflow-wrap: break-word;} /* breakline for metric text         */div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {overflow-wrap: break-word;white-space: break-spaces;color: green;font-size: 20px;} </style>""", unsafe_allow_html=True)

        with precision:
            st.metric(label="Precision Score", value="79.6%")
        with recall:
            st.metric(label="Recall Score", value="79.6%")
    
        with f1:
            st.metric(label="F1 Score", value="79.6%")
        with acc:
            st.metric(label="Accuracy Score", value="79.6%")



main()