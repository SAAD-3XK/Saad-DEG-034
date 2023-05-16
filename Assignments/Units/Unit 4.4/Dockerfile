FROM python:3.8-slim-buster

WORKDIR /home/app/

COPY ./ /home/app/

#moved the requirements.txt installation from run time to build time
RUN pip install -r requirements.txt 


ENV PYTHONPATH=${PYTHONPATH}:/home/app/

#specified the volume where the data will be stored
VOLUME /home/app/vol 


# here we have specified LOG_LEVEL which can be set in the docker-compose file
CMD ["bash", "-c", "gunicorn main3:app -b 0.0.0.0:5000 --log-level=$LOG_LEVEL"]
