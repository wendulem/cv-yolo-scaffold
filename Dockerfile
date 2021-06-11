FROM tiangolo/uwsgi-nginx-flask:python3.8

# the port number need to be incremented as each new project is added
# edit the two lines below
ENV LISTEN_PORT 9000
EXPOSE 9000

COPY ./app /app

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt