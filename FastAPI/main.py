from fastapi import Depends, FastAPI, HTTPException, status, Request
from datetime import timedelta
import numpy as np
import pandas as pd
import pickle
import math
import datetime
import sklearn
from sklearn import preprocessing

app=FastAPI()

def haversian_distance(lat1, lat2, lon1,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

def weather_fields(month:int):
    weather_findings=pd.read_csv('weather_findings.csv')
    df=weather_findings[weather_findings['pickup_month']==month]
    return df

@app.post("/predict", tags=["Predict"])
async def predict(request: Request):
    # model
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    features = await request.json()
    
    pickup_datetime= features["pickup_datetime"]
    pickup_latitude= features["pickup_latitude"]
    pickup_longitude= features["pickup_longitude"]
    dropoff_latitude= features["dropoff_latitude"]
    dropoff_longitude= features["dropoff_longitude"]
    passenger_count= features["passenger_count"]

    to_predict = [pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,pickup_datetime]

    test = pd.DataFrame(data=[to_predict],columns=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude', 'passenger_count','pickup_datetime'])
    test = test.astype({'pickup_datetime':'datetime64'})
    test['pickup_day'] = test.pickup_datetime.dt.day
    test['pickup_month'] = test.pickup_datetime.dt.month
    test['pickup_year'] = test.pickup_datetime.dt.year
    test['pickup_hour'] = test.pickup_datetime.dt.hour
    test['hav_distance'] = test.apply(lambda row:haversian_distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)
    
    my_month=datetime.datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S").month
    df=weather_fields(my_month)
    test['avg_wind'] = int(df.iloc[0,1])
    test['max_temp'] = int(df.iloc[0,2])
    test['min_temp'] = int(df.iloc[0,3])
    test['precipitation'] = int(df.iloc[0,4])
    test['snow_depth'] = int(df.iloc[0,5])
    test['snowfall'] = int(df.iloc[0,6])

    test.drop(columns=['pickup_datetime'],inplace=True)
    
    prediction = model.predict(test)
    return {'Fare':float(prediction)}