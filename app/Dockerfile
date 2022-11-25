# Use and existing docker image as a base
FROM python:3.9.2-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir /app

COPY . /app

WORKDIR /app

RUN pip3 install -r ./requirements.txt

EXPOSE 8501

#ENTRYPOINT ["streamlit","run"]

CMD ["streamlit","run","app.py"]
# CMD ["python" "./app.py"]