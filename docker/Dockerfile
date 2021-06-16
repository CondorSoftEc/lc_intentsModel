FROM python:3.8.10-slim-buster

WORKDIR /app

COPY deployReq.txt requirements.txt

RUN pip3 install -r requirements.txt && pip install jupyter
COPY . .

EXPOSE 8888

# TODO: establecer un volumen para guardar datos

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
