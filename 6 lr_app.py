import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle


model = pickle.load(open('linear_regression_model.pkl','rb'))

# let's create web app
st.title("Scikit-learn Linear Regression model")
tv = st.text_input('Enter TV sales...')
radio = st.text_input("Enter radio sales....")
newspaper = st.text_input("Enter newspaper sales.....")

if st.button("predict"):
    features = np.array([[tv,radio,newspaper]], dtype=np.float64)
    results = model.predict(features).reshape(1,-1)
    st.write("Predicted sale::::", results[0])