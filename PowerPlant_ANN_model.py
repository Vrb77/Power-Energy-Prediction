import streamlit as st
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
# set the tab title
st.set_page_config("Power Energy Prediction")

# Set the page title
st.title("Power Energy Prediction Project")

# Set header
st.subheader("By Vaishnavi Badade")

# Load the pipeline (data cleaning, preprocessing) and model
pre = joblib.load("PowerPlant_model_pre.joblib")
model = load_model("PowerPlant_model.keras")



AT = st.number_input("Ambient Temperature")
V =st.number_input("Exhaust Vacuum")
AP = st.number_input("Ambient Pressure")
RH	= st.number_input("Relative Humidity")


# Include a button. After providing all the inputs, user will click on the button. The button should provide the necessary predictions
submit = st.button("Power Energy Prediction")

if submit:
    data = {
        'AT':[AT],
        'V':[V],
        'AP':[AP],
        'RH':[RH]
    }
    # Convert above dictionary into dataframe first
    xnew = pd.DataFrame(data)
    # Apply data cleaning and preprocessing on new data using pre pipeline
    xnew_pre = pre.transform(xnew)
    # predictions
    preds = model.predict(xnew_pre)
   
    st.subheader(preds)