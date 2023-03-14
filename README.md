[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![DockerHub](https://img.shields.io/badge/DockerHub-0db7ed?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)
[![GCP](https://img.shields.io/badge/GCP-4285F4?style=flat-square&logo=google-cloud&logoColor=white)](https://cloud.google.com/)


# **NewYork taxi fare prediction**
---------------------------------------

The objective of this project is to predict the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. We can get a basic estimate based on just the distance between the two points, this will result in an RMSE of $5-$8. Hence, our aim is to do better than this by using Machine Learning techniques. Meanwhile, also analyzing the rides data to gain some useful insights about the numerous factors which might affect the ride fare.

```
📦 NewYork-taxi-fare-prediction
├─ .gitignore
├─ Data wrangling.ipynb
├─ Demo.gif
├─ Exploratory Data Analysis.ipynb
├─ FastAPI
│  ├─ Dockerfile
│  ├─ main.py
│  ├─ requirements.txt
│  ├─ weather_findings.csv
│  └─ xgb_model.pkl
├─ Feature Engineering.ipynb
├─ Model development.ipynb
├─ NYC_Taxi_Fare_Prediction.mp4
├─ README.md
├─ Streamlit
│  ├─ Dockerfile
│  ├─ demo.py
│  └─ requirements.txt
├─ docker-compose.yml
├─ env_nyc
│  ├─ Scripts
│  │  ├─ Activate.ps1
│  │  ├─ activate
│  │  ├─ activate.bat
│  │  ├─ deactivate.bat
│  │  ├─ easy_install-3.9.exe
│  │  ├─ easy_install.exe
│  │  ├─ f2py.exe
│  │  ├─ gunicorn.exe
│  │  ├─ jsonschema.exe
│  │  ├─ markdown-it.exe
│  │  ├─ normalizer.exe
│  │  ├─ pip.exe
│  │  ├─ pip3.9.exe
│  │  ├─ pip3.exe
│  │  ├─ plasma_store.exe
│  │  ├─ pygmentize.exe
│  │  ├─ pysemver.exe
│  │  ├─ python.exe
│  │  ├─ pythonw.exe
│  │  ├─ streamlit.cmd
│  │  ├─ streamlit.exe
│  │  ├─ uvicorn.exe
│  │  └─ watchmedo.exe
│  ├─ etc
│  │  └─ jupyter
│  │     └─ nbconfig
│  │        └─ notebook.d
│  │           └─ pydeck.json
│  ├─ pyvenv.cfg
│  ├─ share
│  │  └─ jupyter
│  │     └─ nbextensions
│  │        └─ pydeck
│  │           ├─ extensionRequires.js
│  │           ├─ index.js
│  │           └─ index.js.map
│  └─ xgboost
│     └─ vcomp140.dll
└─ requirements.txt
```

### Description of Dataset:- (https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)
The given dataset consists of 55 million entries. I decided to proceed with 2 million data entries, as it would be sufficient for training purposes, though we can take any sample of data but 55 million is too large. The entire dataset can be broadly categorized into the following:
1. train.csv - Input features and target fare_amount values for the training set (about 55M rows).
2. test.csv - Input features for the test set (about 10K rows). Our goal is to predict fare_amount for each row.
3. sample_submission.csv - a sample submission file in the correct format (columns key and fare_amount). This file 'predicts' fare_amount to be $11.35 for all rows, which is the mean fare_amount from the training set.

### Methodology:-
Since, we have to predict the taxi ride fare amount, given certain features, this becomes a regression problem. We already have the taxi ride data obtained from Kaggle. We would also leverage the weather data for NewYork from Jan 1st 2009 to Nov 11th 2015. Weather dataset is obtained from https://www.ncdc.noaa.gov/cdo-web/datasets which provides daily summaries of weather for specific cities and time periods. Features like Snowfall, Snow Depth, Min/Max Temperature, Precipitation, and Air Wind, etc., could act as deterministic factors for ride availability and fare amount. I followed below processes to achieve our goal of predicting the fare amount in an efficient manner:-
1.	Data cleaning and pre-processing
2.	Exploratory Data Analysis
3.	Feature engineering
4.	ML algorithms

I have used below machine learning models to solve the regression problem:
1.	Linear regression
2.	Decision tree regressor
3.	Random Forest regressor
4.	XGBoost regressor
5.	Light GBM regressor

Below are the cumulative performance metrics scores for all the employed ML models:-
</br>
<img width="238" alt="image" src="https://user-images.githubusercontent.com/108916132/225148087-363b7795-262a-4f04-9d08-9213a056282d.png">

Careful and precise analysis was performed over all the aspects of these models. I compared all the performance metrics scores of these models and analyzed the significance of each. Post introspection of all these models, I can narrow down to ‘XGBoost regressor’ as the best performing model for our use-case. The RMSE score for this model is the lowest amongst all trained models. Also, the R2 score which demonstrates aacuracy of a model, is the highest. So, I can confidently quote that this model will generate the most efficient predictions in determining the taxi ride fare amount for a future ride in NewYork state.

### Deployment:-
![deployment_architecture_diagram](https://user-images.githubusercontent.com/108916132/225142720-a25157ae-3f83-4c89-a2bf-24f0c1b898f8.png)

1. Designed an API using FastAPI framework to expose the XGBoost model funtionalities.
2. Designed front-end web-application using Streamlit package to interact with the FastAPI endpoints.
3. Containerized both the applications using Docker containers, allowing for easier sharing and scalability. Published the Docker images on DockerHub, making it easily accessible to others over the internet (https://hub.docker.com/r/mittal15/streamlit_nyc/tags, https://hub.docker.com/r/mittal15/fastapi_nyc/tags).
4. Deployed the application on a Google Cloud Platform (GCP) VM instance through a docker compose file, utilizing top-tier cloud computing infrastructure to provide fast and reliable hosting.

### Application demo:-
![Demo](https://user-images.githubusercontent.com/108916132/225139532-0003b6aa-475c-4f26-82bc-7f903a15e96e.gif)

### Link to full explanatory video:-
[![Video thumbnail](https://user-images.githubusercontent.com/108916132/225133491-0c709a03-1600-47d2-963b-9a29d85e3304.png)](https://github.com/Hmittal15/NewYork-taxi-fare-prediction/raw/main/NYC_Taxi_Fare_Prediction.mp4 "Download or view the video")

### Future Work:-
1.	We can leverage Google maps API and use JavaScript to make the frontend web application more interactive.

## You can find me on <a href="http://www.linkedin.com/in/harshit-mittal-52b292131"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/768px-LinkedIn_logo_initials.png" width="17" height="17" /></a>
