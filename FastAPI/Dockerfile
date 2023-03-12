FROM python:3.9.0

# Pull the base docker image of python with tag 3.9.0

RUN pip install --upgrade pip

WORKDIR /app
# Change the working dir inside the container - cd /app

ADD main.py requirements.txt xgb_model.pkl weather_findings.csv /app/

RUN pip install -r requirements.txt
# Install the dependency

### Container Env same as local at this point

EXPOSE 8090

CMD ["gunicorn" ,"-w", "4", "-k", "uvicorn.workers.UvicornWorker" , "--bind", "0.0.0.0:8090", "main:app"]