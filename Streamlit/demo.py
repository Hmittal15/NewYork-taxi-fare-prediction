from operator import concat
import streamlit as st
import requests
import datetime

def main():
    BASE_URL = "http://fastapi:8090"
    st.set_page_config(page_title="NYC Taxi Fare", page_icon="🚕")
    st.title("Taxi Fare Prediction")
    st.header('Welcome to New York Taxi service!')
    st.write('Enter the trip details to get fare estimate 🚕')
    with st.form("my_form"):
        
        # pickup_datetime = st.text_input('Pickup Datetime',value="2015-01-27 13:08:24 UTC")
        pickup_latitude = st.number_input('Pickup Latitude',value=40.5,min_value=40.477399,max_value=40.917577)
        pickup_longitude = st.number_input('Pickup Longitude',value=-74.0,min_value=-74.259090,max_value=-73.700272)
        dropoff_latitude = st.number_input('Dropoff Latitude',value=40.5,min_value=40.477399,max_value=40.917577)
        dropoff_longitude = st.number_input('Dropoff Longitude',value=-74.0,min_value=-74.259090,max_value=-73.700272)
        passenger_count = st.number_input('Passenger Count',min_value=1,max_value=6)

        features = {
                "pickup_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pickup_latitude" : pickup_latitude,
                "pickup_longitude" : pickup_longitude,
                "dropoff_latitude" : dropoff_latitude,
                "dropoff_longitude" : dropoff_longitude,
                "passenger_count" : passenger_count
            }
            
        
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            response = requests.post(url=concat(BASE_URL, "/predict"), json=features).json()
            if response:
                st.metric(label="Predicted Taxi Fare",value=round(response['Fare'],2))
            else:
                st.error("Error")


if __name__ == '__main__':
    main()