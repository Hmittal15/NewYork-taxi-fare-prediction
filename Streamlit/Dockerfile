FROM python:3.9.0

# Pull the base docker image of python with tag 3.9.0

RUN pip install --upgrade pip

WORKDIR /app
# Change the working dir inside the container - cd /app

ADD demo.py requirements.txt /app/

RUN pip install -r requirements.txt
# Install the dependency

RUN pip install streamlit --upgrade

### Container Env same as local at this point

EXPOSE 8000

CMD ["streamlit", "run", "demo.py", "--server.port", "8000"]