import streamlit as st
import pickle
import numpy as np
with open(r"C:\Users\anbud\Sample\MLprojects\classifier.pkl", "rb") as model_file:
  pipmodel = pickle.load(model_file)
st.title("iris pedictor")
st.write("Enter the following details:")
s_l = float(st.number_input("sepal length", min_value=1, max_value=5))
s_w = float(st.number_input("sepal width", min_value=1, max_value=5))
p_l = float(st.number_input("petal length", min_value=1, max_value=5))
p_w = float(st.number_input("petal width", min_value=1, max_value=5))
if st.button("Predict"):
 features = np.array([[s_l,s_w,p_l,p_w]])
 prediction = pipmodel.predict(features)
 st.write(f"flower: {prediction[0]}")