# TODO: Use alpine for smaller image size
FROM python:3
ENV PYTHONUNBUFFERED 1

RUN mkdir /code
COPY . /code

WORKDIR /code/webapp

RUN pip install -r requirements.txt
EXPOSE 8080

# Start gunicorn
CMD ["../start.sh"]
