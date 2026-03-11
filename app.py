import streamlit as st
from src.prediction import insurance

st.title("Insurance Prediction")
st.write("This is a simple insurance prediction app.")

Age=st.number_input("Enter age: ")
Annual_Income_LPA=st.number_input("Enter annual income (LPA): ")
Policy_Term_Years=st.number_input("Enter policy term_years: ")
Sum_Assured_Lakhs=st.number_input("Enter sum assured_Lakhs: ")

if st.button("Predict"):
    model=insurance()
    result=model.prediction1(Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs)
    st.success(result)
    st.snow()

