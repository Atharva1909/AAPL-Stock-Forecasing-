import streamlit as st
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime


st.title("Apple_Stock_Exchange_Forecast_by_Group$4")
st.subheader('===================================================================')
st.header('Actionable 30-day price forecasts to guide investment strategies for future stock.')
st.subheader('===================================================================')
st.write("Here i tried to demostrate model deployment part by use of streamlit")

st.markdown('[Refrence website for Prophet](https://www.analyticsvidhya.com/blog/2018/05/generate-accurate-forecasts-facebook-prophet-python-r/)')

st.sidebar.title("Time Series Forecasting")
st.sidebar.write("This project forecasts Apple stock prices using Prophet and TES models.")

st.sidebar.write("-----------")
# Display today's date and time
current_date = datetime.now().strftime("%d-%m-%Y")
current_time = datetime.now().strftime("%H:%M")
st.sidebar.write(f"Today's Date: {current_date}")
st.sidebar.write("-----------")
st.sidebar.write(f"Current Time: {current_time}")



# Title of the app
st.subheader('===================================================================')
st.subheader('Select the Model and Forecast the Price')
# Checkbox for terms and conditions
if st.checkbox("I agree to the terms and conditions"):
    
    # Model selection
    model_choice = st.selectbox("Select the forecasting model", ["Prophet", "TES"])
    
    # Slider for forecast steps
    steps = st.slider("Number of steps to forecast", min_value=1, max_value=31, value=12)

    # Button to start the forecast
    if st.button('Start Forecast'):
        if model_choice == "Prophet":
            model_file = 'pro_model.pkl'
        else:
            model_file = 'model.pkl'
        
        # Load the chosen model
        with open(model_file, 'rb') as f:
            my_model = pkl.load(f)

        # Generate the forecast
        if model_choice == "Prophet":
            future = my_model.make_future_dataframe(periods=steps)
            forecast = my_model.predict(future)
            forecast_values = forecast[['ds', 'yhat']].tail(steps)
            forecast_values.columns = ['Date', 'Forecasted Value']
            st.write(forecast_values)
            fig = px.line(forecast_values, x=forecast_values["Date"], y='Forecasted Value', title='Forecasted Stock Prices' )
            st.plotly_chart(fig)	
        else:
            forecast_values = my_model.forecast(steps=steps)
            forecast_values = pd.DataFrame(forecast_values, columns=['Forecasted Value'])
            st.write(forecast_values)
            fig = px.line(forecast_values, x=forecast_values.index, y='Forecasted Value', title='Forecasted Stock Prices' )
            st.plotly_chart(fig)


	
        
